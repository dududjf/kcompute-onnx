import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class ConcatFromSequenceOp:
    def __init__(self, manager: kp.Manager, axis: int = 0, new_axis: int = 0):
        self.manager = manager
        self.axis = axis
        self.new_axis = new_axis
        self.compile_shader = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout(std430, set=0, binding=0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout(std430, set=0, binding=2) readonly  buffer UIParams {{ uint params[]; }};

void main() 
{{
    uint group_count = params[0], axis_dim = params[1], block_size = params[2];
    uint out_axis_offset = params[3], out_axis_dim = params[4];

    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    if (gx >= group_count || gy >= axis_dim) return;

    uint in_offset = gx * axis_dim * block_size + gy * block_size;
    uint out_offset = gx * out_axis_dim * block_size + (out_axis_offset + gy) * block_size;

    for (uint i = 0; i < block_size; ++i, ++in_offset, ++out_offset) {{
        out_tensor[out_offset] = in_tensor[in_offset];
    }}
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ConcatFromSequenceOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs[0]:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors

        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        axis = self.axis
        new_axis = self.new_axis
        rank = len(input_tensors[0][1])

        if new_axis == 1:
            axis = axis + rank + 1 if axis < 0 else axis
            for idx, (tensor, shape) in enumerate(input_tensors):
                tensor_shape = shape[:axis] + [1] + shape[axis:]
                input_tensors[idx] = (tensor, tensor_shape)
        else:
            axis = axis + rank if axis < 0 else axis

        shape_out = input_tensors[0][1].copy()
        shape_out[axis] = sum(shape[axis] for _, shape in input_tensors)

        tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
        updated_tensors.append(tensor_out)

        group_count = int(np.prod(shape_out[:axis])) if axis > 0 else 1
        block_size = int(np.prod(shape_out[axis + 1:])) if axis + 1 < len(shape_out) else 1

        offset = 0
        for tensor, shape in input_tensors:
            axis_dim = shape[axis]
            params = np.array([group_count, axis_dim, block_size, offset, shape_out[axis]], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            group_x = (group_count + LOCAL_X_2D - 1) // LOCAL_X_2D
            group_y = (axis_dim + LOCAL_Y_2D - 1) // LOCAL_Y_2D
            workgroup = (group_x, group_y, 1)

            alg = self.manager.algorithm([tensor, tensor_out, param_in], self.compile_shader, workgroup)
            updated_algorithms.append(alg)
            offset += axis_dim

        return [(tensor_out, shape_out)]

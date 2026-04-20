import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class ArgMaxOp:
    def __init__(self, manager: kp.Manager, axis=0, keepdims=True, select_last_index=False):
        self.manager = manager
        self.axis = axis
        self.keepdims = keepdims
        self.select_last_index = select_last_index
        self.compiled_shader = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout(std430, set = 0, binding = 0) readonly   buffer InBuf  {{ float in_tensor[];  }};
layout(std430, set = 0, binding = 1) writeonly  buffer OutBuf {{ uint out_tensor[];  }};
layout(std430, set = 0, binding = 2) readonly   buffer UIParam {{ uint params[]; }};

void main() {{
    uint batch_size = params[0], block_size = params[1], axis_size = params[2], select_last = params[3];
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    if(gx >= batch_size || gy >= block_size) return;

    uint base_idx = gx * axis_size * block_size + gy;

    float max_val = in_tensor[base_idx];
    uint max_idx = 0;
    base_idx += block_size;

    if (select_last == 1) {{
        for (uint i = 1; i < axis_size; ++i, base_idx += block_size) {{
            if (in_tensor[base_idx] >= max_val) {{
                max_val = in_tensor[base_idx];
                max_idx = i;
            }}
        }}
    }}
    else {{
        for (uint i = 1; i < axis_size; ++i, base_idx += block_size) {{
            if (in_tensor[base_idx] > max_val) {{
                max_val = in_tensor[base_idx];
                max_idx = i;
            }}
        }}
    }}
    out_tensor[gx * block_size + gy] = max_idx;
}}
""")

    def __repr__(self):
        return f"ArgMaxOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape).astype(np.int64)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        axis = self.axis

        axis += len(shape_in) if axis < 0 else 0

        axis_size = shape_in[axis]
        batch_size = int(np.prod(shape_in[:axis])) if axis >= 0 else 1
        block_size = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1

        tensor_out = self.manager.tensor_t(np.zeros(np.prod(batch_size * block_size), dtype=np.int32))
        updated_tensors.append(tensor_out)

        param = [batch_size, block_size, axis_size, self.select_last_index]
        param_in = self.manager.tensor_t(np.array(param, dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        group_x = (batch_size + LOCAL_X_2D - 1) // LOCAL_X_2D
        group_y = (block_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D
        workgroup = (group_x, group_y, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out, param_in],
            self.compiled_shader,
            workgroup
        ))

        if self.keepdims:
            output_shape = shape_in[:axis] + [1] + shape_in[axis + 1:]
        else:
            output_shape = shape_in[:axis] + shape_in[axis + 1:]

        return [(tensor_out, output_shape)]

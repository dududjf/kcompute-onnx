import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class ReduceSumOp:

    def __init__(self, manager: kp.Manager, keepdims=True, noop_with_empty_axes=False):
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes
        self.manager = manager
        self.compiled_shader = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{
    uint dimension;
    uint block_size;
    uint bound_x;
    uint bound_y;
}};

void main()
{{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    if (gx >= bound_x || gy >= bound_y) return;

    uint in_offset  = gx * dimension * block_size + gy;
    uint out_offset = gx * block_size + gy;

    float acc = 0.0;
    for (uint i = 0u; i < dimension; ++i, in_offset += block_size)
        acc += in_tensor[in_offset];

    out_tensor[out_offset] = acc;
}}
""")

    def __repr__(self):
        return f"ReduceSumOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is not None:
                numpy_in = inp.reshape(-1).astype(np.float32) \
                    if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            if tensor is not None:
                del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        axes = input_tensors[1][0].data().astype(int) if len(input_tensors) > 1 else None

        if self.noop_with_empty_axes and axes is None:
            return [input_tensors[0]]

        tensor_in = input_tensors[0][0]
        shape_in = input_tensors[0][1]
        axis_present = [True if axes is None else False] * len(shape_in)
        if axes is not None:
            for axis in axes:
                if axis >= 0:
                    axis_present[axis] = True
                else:
                    axis_present[axis + len(shape_in)] = True

        tensor_out = tensor_in
        block_size = 1

        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i >= 0 else 1

                # 创建参数张量并同步到GPU
                params = [shape_in[i], block_size, group_x, block_size]
                param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                # 计算工作组数量
                workgroup_x = (group_x + LOCAL_X_2D - 1) // LOCAL_X_2D
                workgroup_y = (block_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D
                workgroup = (workgroup_x, workgroup_y, 1)

                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(numpy_out)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_out, param_in],
                    self.compiled_shader,
                    workgroup
                ))
                updated_tensors.append(tensor_out)
            else:
                block_size *= int(shape_in[i])

        if self.keepdims:
            shape_out = [1 if axis_present[i] else shape_in[i] for i in range(len(shape_in))]
        else:
            shape_out = [shape_in[i] for i in range(len(shape_in)) if not axis_present[i]]

        return [(tensor_out, shape_out)]


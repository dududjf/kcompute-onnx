import kp
import numpy as np
from .shader_utils import compile_source


class ReduceLogSumOp:
    def __init__(self, manager: kp.Manager, keepdims=True, noop_with_empty_axes=False):
        self.manager = manager
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes
        self.compiled_shader_sum = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) buffer buf_in_tensor { float in_tensor[]; };
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float dimension_f = 0;
layout (constant_id = 1) const float block_size_f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);

    uint in_offset = gx * dimension * block_size + gy;
    uint out_offset = gx * block_size + gy;

    float acc = 0;
    for(uint i = 0; i < dimension; ++i, in_offset += block_size) {
        acc += in_tensor[in_offset];
    }
    out_tensor[out_offset] = acc;
}
""")

        self.compiled_shader_log = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) buffer buf_in_tensor { float in_tensor[]; };
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; };

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    out_tensor[gx] = log(in_tensor[gx]);
}
""")

    def __repr__(self):
        return f"ReduceLogSumOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        if inputs[0].size == 0:
            input_tensors.append((None, []))
        else:
            input_tensors.append((self.manager.tensor(inputs[0]), list(inputs[0].shape)))
        if len(inputs) > 1:
            if inputs[1] is not None:
                numpy_in = inputs[1].reshape(-1).astype(np.float32) \
                    if isinstance(inputs[1], np.ndarray) else np.array(inputs[1], dtype=np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inputs[1].shape) if isinstance(inputs[1], np.ndarray) else [len(inputs[1])]))

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

        if tensor_in is None:
            shape_out = [1]
            numpy_out = np.full(int(np.prod(shape_out)), -np.inf, dtype=np.float32)
            tensor_out = self.manager.tensor(numpy_out)
            updated_tensors.append(tensor_out)
            return [(tensor_out, shape_out)]

        if axes is None:
            axis_present = [True] * len(shape_in)
        else:
            axis_present = [False] * len(shape_in)
            for axis in axes:
                idx = axis if axis >= 0 else axis + len(shape_in)
                axis_present[idx] = True

        tensor_out = tensor_in
        block_size = 1

        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i > 0 else 1
                workgroup = (group_x, block_size, 1)
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(numpy_out)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_out],
                    self.compiled_shader_sum,
                    workgroup,
                    [shape_in[i], block_size],
                    []
                ))
                updated_tensors.append(tensor_out)
            else:
                block_size *= int(shape_in[i])

        tensor_out_log = tensor_out
        updated_tensors.append(tensor_out_log)
        updated_algorithms.append(self.manager.algorithm([tensor_out, tensor_out_log], self.compiled_shader_log))

        if self.keepdims:
            shape_out = [1 if axis_present[i] else shape_in[i] for i in range(len(shape_in))]
        else:
            shape_out = [shape_in[i] for i in range(len(shape_in)) if not axis_present[i]]

        return [(tensor_out_log, shape_out)]
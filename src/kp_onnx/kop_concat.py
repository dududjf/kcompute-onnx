import numpy as np
import kp
from .shader_utils import compile_source


class ConcatOp:
    def __init__(self, manager: kp.Manager, axis: int = 0):
        self.manager = manager
        self.axis = axis
        self.copy_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf  { float in_tensor[];  };
layout(binding = 1) writeonly buffer out_buf { float out_tensor[]; };

layout(constant_id = 0) const float axis_dim_f = 0.0;
layout(constant_id = 1) const float block_size_f = 0.0;
layout(constant_id = 2) const float out_axis_offset_f = 0.0;
layout(constant_id = 3) const float out_axis_dim_f = 0.0;
layout(constant_id = 4) const float group_count_f = 0.0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint axis_dim = uint(axis_dim_f);
    uint block_size = uint(block_size_f);
    uint out_axis_offset = uint(out_axis_offset_f);
    uint out_axis_dim = uint(out_axis_dim_f);
    uint group_count = uint(group_count_f);

    if (gx >= group_count || gy >= axis_dim) return;

    uint in_offset = gx * axis_dim * block_size + gy * block_size;
    uint out_offset = gx * out_axis_dim * block_size + (out_axis_offset + gy) * block_size;

    for (uint i = 0u; i < block_size; ++i, ++in_offset, ++out_offset) {
        out_tensor[out_offset] = in_tensor[in_offset];
    }
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ConcatOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ConcatOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        for arr in inputs:
            numpy_in = arr.reshape(-1).astype(np.float32) \
                if isinstance(arr, np.ndarray) else np.array(arr, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(arr.shape) if isinstance(arr, np.ndarray) else []))

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
        for idx, (tensor, shape) in enumerate(input_tensors):
            rank = len(shape)
            if axis >= 0 and axis >= rank:
                tensor_shape = shape.copy()
                tensor_shape += [1] * (axis + 1 - rank)
                input_tensors[idx] = (tensor, tensor_shape)
            else:
                input_tensors[idx] = (tensor, shape.copy())

        base_shape = input_tensors[0][1]
        adj_axis = axis + len(base_shape) if axis < 0 else axis

        if not (0 <= adj_axis < len(base_shape)):
            raise RuntimeError(f"ConcatOp: computed concat axis {adj_axis} out of range for base shape {base_shape}")

        for _, shape in input_tensors[1:]:
            if len(shape) != len(base_shape):
                raise RuntimeError(
                    f"ConcatOp: input shapes must have same rank after preprocessing, got {shape} vs {base_shape}")
            for i, (s0, s1) in enumerate(zip(base_shape, shape)):
                if i != adj_axis and s0 != s1:
                    raise RuntimeError(
                        f"ConcatOp: all dimensions except concat axis must match, mismatch at dim {i}: {s0} vs {s1}")

        tensor_shape = base_shape.copy()
        tensor_shape[adj_axis] = sum(shape[adj_axis] for _, shape in input_tensors)
        out_size = int(np.prod(tensor_shape))

        out_array = np.zeros(out_size, dtype=np.float32)
        tensor_out = self.manager.tensor(out_array)
        updated_tensors.append(tensor_out)

        group_count = int(np.prod(base_shape[:adj_axis])) if adj_axis > 0 else 1
        block_size = int(np.prod(base_shape[adj_axis + 1:])) if adj_axis + 1 < len(base_shape) else 1
        out_axis_dim = tensor_shape[adj_axis]

        offset = 0
        for tensor, shape in input_tensors:
            axis_dim = shape[adj_axis]
            workgroup = (group_count, axis_dim, 1)
            constants = [axis_dim, block_size, offset, out_axis_dim, group_count]
            alg = self.manager.algorithm([tensor, tensor_out], self.copy_shader, workgroup, constants, [])
            updated_algorithms.append(alg)
            offset += axis_dim

        return [(tensor_out, tensor_shape)]

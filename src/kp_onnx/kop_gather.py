import numpy as np
import kp
from .shader_utils import compile_source


class GatherOp:
    def __init__(self, manager: kp.Manager, axis: int = 0):
        self.manager = manager
        self.axis = axis
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) readonly  buffer indices_buf{ int   indices[];       };
layout(binding = 2) writeonly buffer out_buf    { float out_tensor[];    };

layout(constant_id = 0) const float axis_dim_f = 0;
layout(constant_id = 1) const float block_size_f = 0;
layout(constant_id = 2) const float out_block_size_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint axis_dim       = uint(axis_dim_f);
    uint block_size     = uint(block_size_f);
    uint out_block_size = uint(out_block_size_f);

    int idx = indices[gy];
    if (idx < 0) idx += int(axis_dim);

    uint in_offset  = gx * axis_dim * block_size + uint(idx) * block_size;
    uint out_offset = gx * out_block_size + gy * block_size;

    for (uint i = 0; i < block_size; i++, in_offset++, out_offset++) {
        out_tensor[out_offset] = in_tensor[in_offset];
    }
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GatherOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GatherOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "GatherOp requires two inputs: data and indices"
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape) if isinstance(inputs[0], np.ndarray) else []))

        numpy_indices = np.array(inputs[1], dtype=np.int32).reshape(-1)
        tensor_indices = self.manager.tensor_t(numpy_indices, kp.TensorTypes.device)
        input_tensors.append((tensor_indices, list(inputs[1].shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
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
        tensor_x, shape_x = input_tensors[0]
        tensor_indices, shape_indices = input_tensors[1]

        axis = self.axis
        if axis < 0:
            axis += len(shape_x)

        axis_dim = shape_x[axis]
        group_count = int(np.prod(shape_x[:axis])) if axis > 0 else 1
        block_size = int(np.prod(shape_x[axis + 1:])) if axis + 1 < len(shape_x) else 1

        indices_data = tensor_indices.data().astype(np.int32)
        if indices_data.size == 0:
            tensor_shape = shape_x[:axis] + shape_indices + shape_x[axis + 1:]
            out_array = np.empty(0, dtype=np.float32)
            tensor_out = self.manager.tensor(out_array)
            updated_tensors.append(tensor_out)
            return [(tensor_out, tensor_shape)]

        out_axis_dim = indices_data.size
        tensor_shape = shape_x[:axis] + shape_indices + shape_x[axis + 1:]
        out_size = group_count * out_axis_dim * block_size

        out_array = np.zeros(out_size, dtype=np.float32)
        tensor_out = self.manager.tensor(out_array)
        updated_tensors.append(tensor_out)

        workgroup = (group_count, out_axis_dim, 1)

        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_x, tensor_indices, tensor_out],
                self.compiled_shader,
                workgroup,
                [axis_dim, block_size, out_axis_dim * block_size],
                [],
            )
        )

        return [(tensor_out, tensor_shape)]

import kp
import numpy as np
from .shader_utils import compile_source


class GatherElementsOp:
    def __init__(self, manager: kp.Manager, axis: int = 0):
        self.manager = manager
        self.axis = axis
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly  buffer buf_in_data     { float in_data[];     };
layout(binding = 1) readonly  buffer buf_in_indices  { int   in_indices[];  };
layout(binding = 2) writeonly buffer buf_out_data    { float out_data[];    };
layout(binding = 3) readonly  buffer buf_precalc     { uint  row_offsets[]; };

layout(constant_id = 0) const float axis_dim_f       = 0.0;
layout(constant_id = 1) const float inner_block_f    = 0.0;
layout(constant_id = 2) const float indices_axis_f   = 0.0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint axis_dim    = uint(axis_dim_f);
    uint inner_block = uint(inner_block_f);
    uint indices_axis = uint(indices_axis_f);

    uint idx_ptr = gx * indices_axis * inner_block + gy * inner_block;
    uint in_plane_base = gx * axis_dim * inner_block;
    uint out_base = idx_ptr;

    uint iptr = idx_ptr;
    uint out_ptr = out_base;

    for (uint i = 0u; i < inner_block; ++i) {
        int idx = in_indices[iptr++];
        if (idx < 0) idx += int(axis_dim);

        uint row_base = in_plane_base + row_offsets[uint(idx)];
        out_data[out_ptr++] = in_data[row_base + i];
    }
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GatherElementsOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GatherElementsOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "GatherElementsOp requires two inputs: data and indices"
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape) if isinstance(inputs[0], np.ndarray) else []))

        numpy_in = np.array(inputs[1], dtype=np.int32).reshape(-1)
        tensor = self.manager.tensor_t(numpy_in, kp.TensorTypes.device)
        input_tensors.append((tensor, list(inputs[1].shape)))

        updated_algorithms, updated_tensors = [], []
        tensor_out_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, out_shape = tensor_out_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        output = tensor_out.data().reshape(out_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors

        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        tensor_indices_dev, shape_indices = input_tensors[1]

        axis = self.axis
        if axis < 0:
            axis += len(shape_in)
        axis = max(0, min(axis, len(shape_in) - 1))

        axis_dim = shape_in[axis]
        prefix = int(np.prod(shape_in[:axis])) if axis > 0 else 1
        inner_block = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1
        indices_axis = int(shape_indices[axis]) if axis < len(shape_indices) else 1

        tensor_shape = shape_indices
        out_size = prefix * indices_axis * inner_block
        out_array = np.zeros(out_size, dtype=np.float32)
        tensor_out = self.manager.tensor(out_array)
        updated_tensors.append(tensor_out)

        row_offsets = (np.arange(axis_dim, dtype=np.uint32) * np.uint32(inner_block)).astype(np.uint32)
        tensor_row_offsets = self.manager.tensor_t(row_offsets, kp.TensorTypes.device)
        updated_tensors.append(tensor_row_offsets)

        workgroup = (prefix, indices_axis, 1)
        consts = [axis_dim, inner_block, indices_axis]
        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_indices_dev, tensor_out, tensor_row_offsets],
                self.compiled_shader,
                workgroup,
                consts,
                [],
            )
        )

        return [(tensor_out, tensor_shape)]

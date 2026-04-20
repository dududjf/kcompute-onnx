import numpy as np
import kp
from .shader_utils import compile_source


class FeatureVectorizerOp:
    def __init__(self, manager: kp.Manager, inputdimensions: list[int] = []):
        self.manager = manager
        self.inputdimensions = inputdimensions
        self.compile_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf  { float in_tensor[];  };
layout(binding = 1) writeonly buffer out_buf { float out_tensor[]; };

layout(constant_id = 0) const float input_width_f = 0.0;
layout(constant_id = 1) const float out_axis_offset_f = 0.0;
layout(constant_id = 2) const float out_axis_dim_f = 0.0;

void main() {
    uint row_idx = gl_GlobalInvocationID.x;
    uint col_idx = gl_GlobalInvocationID.y;

    uint input_width = uint(input_width_f);
    uint out_axis_offset = uint(out_axis_offset_f);
    uint out_axis_dim = uint(out_axis_dim_f);

    uint in_offset = row_idx * input_width + col_idx;
    uint out_offset = row_idx * out_axis_dim + out_axis_offset + col_idx;

    out_tensor[out_offset] = col_idx < input_width ? in_tensor[in_offset] : 0.0;
}
""")

    def __repr__(self):
        return f"FeatureVectorizerOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        if tensor_out is not None:
            output = tensor_out.data().reshape(output_shape)
        else:
            output = np.array([], dtype=np.float32)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        batch = input_tensors[0][1][0]
        input_widths = []
        output_widths = []

        for (_, shape), cut in zip(input_tensors, self.inputdimensions):
            assert len(shape) == 1 or len(shape) == 2, f"Every input must have 1 or 2 dimensions not {shape}."
            if len(shape) == 1:
                n_cols = 1
            else:
                n_cols = shape[1]
            input_widths.append(n_cols)

            if cut < 0:
                if cut + n_cols > 0:
                    cut += n_cols
                else:
                    cut = 0
            output_widths.append(cut)

        shape_out = [batch, sum(output_widths)]

        if shape_out[1] == 0:
            return [(None, [])]

        else:
            tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
            updated_tensors.append(tensor_out)

            offset = 0
            for (tensor, shape), input_width, cut_width in zip(input_tensors, input_widths, output_widths):
                if cut_width == 0:
                    continue
                workgroup = (batch, cut_width, 1)
                constants = [input_width, offset, shape_out[1]]
                alg = self.manager.algorithm([tensor, tensor_out], self.compile_shader, workgroup, constants, [])
                updated_algorithms.append(alg)
                offset += cut_width

            return [(tensor_out, shape_out)]
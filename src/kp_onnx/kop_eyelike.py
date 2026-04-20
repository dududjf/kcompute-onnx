import kp
import numpy as np
from .shader_utils import compile_source


class EyeLikeOp:

    def __init__(self, manager: kp.Manager, k=0):
        self.k = k
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) writeonly buffer Out { float out_buf[]; };

layout(constant_id = 0) const float cols_f = 0.0;
layout(constant_id = 1) const float k_f    = 0.0;

void main() {
    uint cols = uint(cols_f);
    int  k    = int(k_f);

    uint r = gl_GlobalInvocationID.y;
    uint c = gl_GlobalInvocationID.x;

    out_buf[r * cols + c] = ((int(c) - int(r)) == k) ? 1.0 : 0.0;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"EyeLikeOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
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
        shape_in = input_tensors[0][1]
        assert 1 <= len(shape_in) <= 2, f"EyeLike only accepts 1D or 2D input, got {shape_in}"

        if len(shape_in) == 1:
            rows = cols = shape_in[0]
        else:
            rows, cols = shape_in
        out_shape = [rows, cols]

        size = rows * cols
        shader = self.shader
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (cols, rows, 1)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_out],
            shader,
            workgroup,
            [cols, self.k],
            []
        ))

        return [(tensor_out, out_shape)]
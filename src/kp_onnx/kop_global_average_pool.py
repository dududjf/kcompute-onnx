import kp
import numpy as np
from .shader_utils import compile_source


class GlobalAveragePoolOp:
    """
    onnx::GlobalAveragePool 的 Kompute 实现（输入形状：N, C, spatial... → 输出形状：N, C, 1, ..., 1 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float S_f = 0.0;
layout (constant_id = 1) const float C_f = 0.0;

void main() {
    uint n = gl_GlobalInvocationID.x;
    uint c = gl_GlobalInvocationID.y;

    uint S = uint(S_f);
    uint C = uint(C_f);
    uint STRIDE_C = S;
    uint STRIDE_N = C * S;

    if (c >= C) { return; }

    uint base = n * STRIDE_N + c * STRIDE_C;

    float acc = 0.0;
    for (uint i = 0u; i < S; ++i) {
        acc += in_buf[base + i];
    }
    float mean = (S > 0u) ? (acc / float(S)) : 0.0;
    out_buf[n * C + c] = mean;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"GlobalAveragePoolOp({dev})"

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

        if updated_algorithms:
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
        tensor_in, in_shape = input_tensors[0]
        assert len(in_shape) >= 2, "GlobalAveragePool expects at least [N, C, ...]"
        if len(in_shape) == 2:
            return [(tensor_in, list(in_shape))]

        N, C = int(in_shape[0]), int(in_shape[1])
        spatial = in_shape[2:]
        S = int(np.prod(spatial, dtype=np.int64))

        out_shape = [N, C] + [1] * len(spatial)
        tensor_out = self.manager.tensor(np.zeros(N * C, dtype=np.float32))
        spec_consts = [float(S), float(C)]
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            (N, C, 1),  # x=N, y=C, z=1
            spec_consts,
            []
        )

        updated_tensors.append(tensor_out)
        updated_algorithms.append(algo)
        return [(tensor_out, out_shape)]

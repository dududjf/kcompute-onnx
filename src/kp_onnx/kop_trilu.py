import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_K = 0


class TriluOp:

    def __init__(self, manager: kp.Manager, upper=1):
        self.upper = upper
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (set = 0, binding = 0) readonly  buffer InBuf  { float in_buf[];  };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float N_f = 0.0;
layout (constant_id = 1) const float M_f = 0.0;
layout (constant_id = 2) const float K_f = 0.0;
layout (constant_id = 3) const float U_f = 1.0;
layout (constant_id = 4) const float B_f = 1.0;

void main() {
    uint j = gl_GlobalInvocationID.x;
    uint i = gl_GlobalInvocationID.y;
    uint b = gl_GlobalInvocationID.z;

    uint N = uint(N_f);
    uint M = uint(M_f);
    int  k = int(K_f);
    bool upper = (U_f > 0.5);
    uint B = uint(B_f);

    uint block = N * M;
    uint idx   = b * block + i * M + j;

    bool keep  = upper ? (int(j) - int(i) >= k)
                       : (int(i) - int(j) >= -k);

    out_buf[idx] = keep ? in_buf[idx] : 0.0;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"TriluOp({dev})"
    __str__ = __repr__

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

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        in_tensor, in_shape = input_tensors[0]
        assert len(in_shape) >= 2, "Trilu expects rank >= 2"
        batch_dims = in_shape[:-2]
        B = int(np.prod(batch_dims)) if batch_dims else 1
        N = in_shape[-2]
        M = in_shape[-1]
        k = input_tensors[1][0].data() if len(input_tensors) >= 2 else DEFAULT_K
        size = np.prod(in_shape)

        out_tensor = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(out_tensor)

        spec_consts = [N, M, k, self.upper, B]
        workgroup = (M, N, B)
        updated_algorithms.append(
            self.manager.algorithm(
                [in_tensor, out_tensor],
                self.shader,
                workgroup,
                spec_consts,
                []
            )
        )
        return [(out_tensor, in_shape)]
import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D

DEFAULT_K = 0


class TriluOp:

    def __init__(self, manager: kp.Manager, upper=1):
        self.upper = upper
        self.manager = manager
        self.shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{
    uint N;
    uint M;
    int  K;
    uint upper;
    uint B;
}};

void main() {{
    uint j = gl_GlobalInvocationID.x;
    uint i = gl_GlobalInvocationID.y;
    uint b = gl_GlobalInvocationID.z;

    if (j >= M || i >= N || b >= B) return;

    uint block = N * M;
    uint idx   = b * block + i * M + j;

    bool keep  = (upper == 1u) ? (int(j) - int(i) >= K)
                                : (int(i) - int(j) >= -K);

    out_buf[idx] = keep ? in_buf[idx] : 0.0;
}}
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
        k = int(input_tensors[1][0].data()) if len(input_tensors) >= 2 else DEFAULT_K
        size = int(np.prod(in_shape))

        out_tensor = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(out_tensor)

        # 创建参数张量并同步到GPU
        params = np.array([N, M, k, self.upper, B], dtype=np.int32).astype(np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        # 计算工作组数量
        workgroup = (
            (M + LOCAL_X_3D - 1) // LOCAL_X_3D,
            (N + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
            (B + LOCAL_Z_3D - 1) // LOCAL_Z_3D
        )

        updated_algorithms.append(
            self.manager.algorithm(
                [in_tensor, out_tensor, param_in],
                self.shader,
                workgroup
            )
        )
        return [(out_tensor, in_shape)]

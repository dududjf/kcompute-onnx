import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class GemmOp:

    def __init__(self, manager: kp.Manager, alpha=1.0, beta=1.0, transA=0, transB=0):
        self.manager = manager
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB
        self.matmul_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly buffer InA {{ float in_a[]; }};
layout (std430, set = 0, binding = 1) readonly buffer InB {{ float in_b[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_buf[]; }};
layout (std430, set = 0, binding = 3) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];
    uint A_cols = params[3];
    uint B_cols = params[4];
    uint transA = params[5];
    uint transB = params[6];
    float alpha = uintBitsToFloat(params[7]);
    
    uint m = gl_GlobalInvocationID.x;
    uint n = gl_GlobalInvocationID.y;
    
    if(m >= M || n >= N) return;
    
    uint a_base = (transA != 0u) ? m : m * A_cols;
    uint a_stride = (transA != 0u) ? A_cols : 1u;
    uint b_base = (transB != 0u) ? n * B_cols : n;
    uint b_stride = (transB != 0u) ? 1u : B_cols;
    
    float sum = 0.0;
    
    uint a_idx = a_base;
    uint b_idx = b_base;
    for (uint k = 0u; k < K; ++k) {{
        sum += in_a[a_idx] * in_b[b_idx];
        a_idx += a_stride;
        b_idx += b_stride;
    }}
    
    out_buf[m * N + n] = sum * alpha;
}}
""")
        self.add_c_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) buffer InOut {{ float in_out[]; }};
layout (std430, set = 0, binding = 1) readonly buffer InC {{ float in_c[]; }};
layout (std430, set = 0, binding = 2) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint M = params[0];
    uint N = params[1];
    uint C_rows = params[2];
    uint C_cols = params[3];
    float beta = uintBitsToFloat(params[4]);
    
    uint m = gl_GlobalInvocationID.x;
    uint n = gl_GlobalInvocationID.y;

    if(m >= M || n >= N) return;

    // Broadcast C to match output shape
    uint c_m = min(m, C_rows - 1u);
    uint c_n = min(n, C_cols - 1u);

    in_out[m * N + n] += in_c[c_m * C_cols + c_n] * beta;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GemmOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, inp.shape))

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
        A_tensor, A_shape = input_tensors[0]
        B_tensor, B_shape = input_tensors[1]

        # ONNX GEMM规范：A和B必须是严格的2D矩阵
        assert len(A_shape) == 2, f"A must be 2D per ONNX spec, got {len(A_shape)}D with shape {A_shape}"
        assert len(B_shape) == 2, f"B must be 2D per ONNX spec, got {len(B_shape)}D with shape {B_shape}"

        # Determine dimensions based on transpose flags
        if self.transA:
            M, K = A_shape[1], A_shape[0]
        else:
            M, K = A_shape[0], A_shape[1]

        if self.transB:
            N, K_check = B_shape[0], B_shape[1]
        else:
            N, K_check = B_shape[1], B_shape[0]

        # Validate K dimension matches
        assert K == K_check, \
            f"Incompatible dimensions for matrix multiplication: " \
            f"A{'T' if self.transA else ''}({M}, {K}) @ B{'T' if self.transB else ''}({K_check}, {N})"

        # Create output tensor
        output_tensor = self.manager.tensor(np.zeros(M * N, dtype=np.float32))

        # Step 1: Matrix multiplication Y = alpha * A' * B'
        params = np.array([M, N, K, A_shape[1], B_shape[1], self.transA, self.transB,
                          np.float32(self.alpha).view(np.uint32)], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        group_x = (M + LOCAL_X_2D - 1) // LOCAL_X_2D
        group_y = (N + LOCAL_Y_2D - 1) // LOCAL_Y_2D

        updated_algorithms.append(
            self.manager.algorithm(
                [A_tensor, B_tensor, output_tensor, param_in],
                self.matmul_shader,
                (group_x, group_y, 1)
            )
        )

        # Step 2: Add C * beta if C is provided and beta != 0
        if len(input_tensors) > 2 and self.beta != 0:
            C_tensor, C_shape = input_tensors[2]

            # ONNX规范：C可以是一维或二维，支持unidirectional broadcasting
            assert len(C_shape) <= 2, f"C must be at most 2D per ONNX spec, got {len(C_shape)}D with shape {C_shape}"

            # 确定C的行列数
            if len(C_shape) == 0:
                C_rows, C_cols = 1, 1
            elif len(C_shape) == 1:
                C_rows, C_cols = 1, C_shape[0]
            else:
                C_rows, C_cols = C_shape[0], C_shape[1]

            # 验证C可以广播到输出形状(M, N)
            assert (C_rows == 1 or C_rows == M) and (C_cols == 1 or C_cols == N), \
                f"Cannot broadcast C with shape {C_shape} to output shape ({M}, {N})"

            params_c = np.array([M, N, C_rows, C_cols, np.float32(self.beta).view(np.uint32)], dtype=np.uint32)
            param_in_c = self.manager.tensor_t(params_c, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in_c])).eval()

            updated_algorithms.append(
                self.manager.algorithm(
                    [output_tensor, C_tensor, param_in_c],
                    self.add_c_shader,
                    (group_x, group_y, 1)
                )
            )

        updated_tensors.append(output_tensor)
        return [(output_tensor, [M, N])]


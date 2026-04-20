import kp
import numpy as np
from .shader_utils import compile_source


class GemmOp:

    def __init__(self, manager: kp.Manager, alpha=1.0, beta=1.0, transA=0, transB=0):
        self.manager = manager
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB
        self.matmul_shader = compile_source(r"""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(set = 0, binding = 0) readonly buffer InA { float in_a[]; };
layout(set = 0, binding = 1) readonly buffer InB { float in_b[]; };
layout(set = 0, binding = 2) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float M_f = 0.0;
layout(constant_id = 1) const float N_f = 0.0;
layout(constant_id = 2) const float K_f = 0.0;
layout(constant_id = 3) const float A_cols_f = 0.0;
layout(constant_id = 4) const float B_cols_f = 0.0;
layout(constant_id = 5) const float alpha = 1.0;
layout(constant_id = 6) const float transA_f = 0.0;
layout(constant_id = 7) const float transB_f = 0.0;

void main() {
    uint m = gl_GlobalInvocationID.x;
    uint n = gl_GlobalInvocationID.y;
    
    uint M = uint(M_f);
    uint N = uint(N_f);
    uint K = uint(K_f);
    uint A_cols = uint(A_cols_f);
    uint B_cols = uint(B_cols_f);
    uint transA = uint(transA_f);
    uint transB = uint(transB_f);
    
    uint a_base = (transA != 0u) ? m : m * A_cols;
    uint a_stride = (transA != 0u) ? A_cols : 1u;
    uint b_base = (transB != 0u) ? n * B_cols : n;
    uint b_stride = (transB != 0u) ? 1u : B_cols;
    
    float sum = 0.0;
    
    uint a_idx = a_base;
    uint b_idx = b_base;
    for (uint k = 0u; k < K; ++k) {
        sum += in_a[a_idx] * in_b[b_idx];
        a_idx += a_stride;
        b_idx += b_stride;
    }
    
    out_buf[m * N + n] = sum * alpha;
}
""")
        self.add_c_shader = compile_source(r"""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer InOut { float in_out[]; };
layout(set = 0, binding = 1) readonly buffer InC { float in_c[]; };

layout(constant_id = 0) const float M_f = 0.0;
layout(constant_id = 1) const float N_f = 0.0;
layout(constant_id = 2) const float C_rows_f = 0.0;
layout(constant_id = 3) const float C_cols_f = 0.0;
layout(constant_id = 4) const float beta = 1.0;

void main() {
    uint m = gl_GlobalInvocationID.x;
    uint n = gl_GlobalInvocationID.y;

    uint M = uint(M_f);
    uint N = uint(N_f);
    uint C_rows = uint(C_rows_f);
    uint C_cols = uint(C_cols_f);

    // Broadcast C to match output shape
    uint c_m = min(m, C_rows - 1u);
    uint c_n = min(n, C_cols - 1u);

    in_out[m * N + n] += in_c[c_m * C_cols + c_n] * beta;
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GemmOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"GemmOp({device_name})"

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
        updated_algorithms.append(
            self.manager.algorithm(
                [A_tensor, B_tensor, output_tensor],
                self.matmul_shader,
                (M, N, 1),
                [M, N, K, A_shape[1], B_shape[1], self.alpha, self.transA, self.transB]
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

            updated_algorithms.append(
                self.manager.algorithm(
                    [output_tensor, C_tensor],
                    self.add_c_shader,
                    (M, N, 1),
                    [M, N, C_rows, C_cols, self.beta]
                )
            )

        updated_tensors.append(output_tensor)
        return [(output_tensor, [M, N])]

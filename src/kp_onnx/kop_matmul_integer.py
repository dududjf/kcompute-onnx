import kp
import numpy as np
from .shader_utils import compile_source


class MatMulIntegerOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader_matmul_2d = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) readonly buffer buf_a { int in_tensor_1[]; };
layout (binding = 1) readonly buffer buf_b { int in_tensor_2[]; };
layout (binding = 2) buffer        buf_c { int C[]; };
layout (constant_id = 0) const float M_f = 0;
layout (constant_id = 1) const float K_f = 0;
layout (constant_id = 2) const float N_f = 0;
void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint M = uint(M_f);
    uint K = uint(K_f);
    uint N = uint(N_f);
    int acc = 0;
    uint a_base = row * K;
    uint b_base = col;
    for (uint k = 0; k < K; ++k, ++a_base, b_base += N) {
        acc += in_tensor_1[a_base] * in_tensor_2[b_base];
    }
    C[row * N + col] = acc;
}
""")
        self.compiled_shader_matmul_batched = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) readonly buffer buf_a { int in_tensor_1[]; };
layout (binding = 1) readonly buffer buf_b { int in_tensor_2[]; };
layout (binding = 2) buffer        buf_c { int C[]; };
layout (constant_id = 0) const float M_f = 0;
layout (constant_id = 1) const float K_f = 0;
layout (constant_id = 2) const float N_f = 0;
layout (constant_id = 3) const float B_f = 0;
void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint bid = gl_GlobalInvocationID.z;
    uint M = uint(M_f);
    uint K = uint(K_f);
    uint N = uint(N_f);
    uint Bn = uint(B_f);
    int acc = 0;
    uint a_base = bid * M * K + row * K;
    uint b_base = bid * K * N + col;
    for (uint k = 0; k < K; ++k, ++a_base, b_base += N) {
        acc += in_tensor_1[a_base] * in_tensor_2[b_base];
    }
    C[bid * M * N + row * N + col] = acc;
}
""")
        # Zero-point preprocessing shaders (no division/modulo broadcasting)
        self.compiled_shader_sub_a = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) readonly buffer buf_in  { int in_tensor[]; };
layout (binding = 1) readonly buffer buf_zp  { int zp[]; };
layout (binding = 2) buffer        buf_out { int out_tensor[]; };
layout (constant_id = 0) const float rows_f = 0;
layout (constant_id = 1) const float cols_f = 0;
layout (constant_id = 2) const float batches_f = 0;
layout (constant_id = 3) const float mode_f = 0; // 0 scalar, 1 per-row, 2 per-batch-row
void main(){
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint bid = gl_GlobalInvocationID.z;
    uint rows = uint(rows_f);
    uint cols = uint(cols_f);
    uint mode = uint(mode_f);
    uint p = (bid * rows + row) * cols + col;
    int v = in_tensor[p];
    int zpv = mode == 1 ? zp[row] : mode == 2 ? zp[bid * rows + row] : zp[0];
    v -= zpv;
    out_tensor[p] = v;
}
""")
        self.compiled_shader_sub_b = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) readonly buffer buf_in  { int in_tensor[]; };
layout (binding = 1) readonly buffer buf_zp  { int zp[]; };
layout (binding = 2) buffer        buf_out { int out_tensor[]; };
layout (constant_id = 0) const float k_size_f = 0;
layout (constant_id = 1) const float num_cols_f = 0;
layout (constant_id = 2) const float batches_f = 0;
layout (constant_id = 3) const float mode_f = 0; // 0 scalar, 1 per-col, 2 per-batch-col
void main(){
    uint k_index = gl_GlobalInvocationID.x;     // along K dimension
    uint col_index = gl_GlobalInvocationID.y;   // along N (columns)
    uint batch_index = gl_GlobalInvocationID.z; // batch index
    uint k_size = uint(k_size_f);
    uint num_cols = uint(num_cols_f);
    uint mode = uint(mode_f);
    uint p = (batch_index * k_size + k_index) * num_cols + col_index;
    int v = in_tensor[p];
    int zpv = mode == 1 ? zp[col_index] : mode == 2 ? zp[batch_index * num_cols + col_index] : zp[0];
    v -= zpv;
    out_tensor[p] = v;
}
""")

    def __repr__(self):
        return f"MatMulIntegerOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.int32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.int32)
            tensor = self.manager.tensor_t(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

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
        tensor_in_1 = input_tensors[0][0]
        tensor_in_2 = input_tensors[1][0]
        shape_1 = input_tensors[0][1]
        shape_2 = input_tensors[1][1]

        # Prepare zero-points (no CPU-side broadcast; keep original tensors)
        provided_azp = False
        provided_bzp = False
        if len(input_tensors) > 2:
            provided_azp = True
            azp_tensor, azp_shape = input_tensors[2]
            azp_size = int(np.prod(azp_shape))
        if len(input_tensors) > 3:
            provided_bzp = True
            bzp_tensor, bzp_shape = input_tensors[3]
            bzp_size = int(np.prod(bzp_shape))

        if len(shape_1) >= 2 and len(shape_2) == 2:
            rows = int(np.prod(shape_1[:-1]))
            cols = shape_1[-1]
            nrows = shape_2[0]
            ncols = shape_2[1]
            assert cols == nrows, f"MatMulIntegerOp: inner dims mismatch {cols} vs {nrows}"

            # Optionally pre-subtract zero-points, then run pure matmul
            a_adj = tensor_in_1
            b_adj = tensor_in_2
            if provided_azp:
                mode_a = 1 if azp_size == rows else 0  # 2D: scalar or per-row
                a_adj = self.manager.tensor_t(np.zeros(rows * cols, dtype=np.int32))
                updated_tensors.append(a_adj)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in_1, azp_tensor, a_adj],
                    self.compiled_shader_sub_a,
                    (rows, cols, 1),
                    [rows, cols, 1, mode_a],
                    []
                ))
            if provided_bzp:
                mode_b = 1 if bzp_size == ncols else 0  # 2D: scalar or per-col
                b_adj = self.manager.tensor_t(np.zeros(cols * ncols, dtype=np.int32))
                updated_tensors.append(b_adj)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in_2, bzp_tensor, b_adj],
                    self.compiled_shader_sub_b,
                    (cols, ncols, 1),
                    [cols, ncols, 1, mode_b],
                    []
                ))

            tensor_out = self.manager.tensor_t(np.zeros(rows * ncols, dtype=np.int32))
            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm(
                [a_adj, b_adj, tensor_out],
                self.compiled_shader_matmul_2d,
                (rows, ncols, 1),
                [rows, cols, ncols],
                []
            ))
            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]

        # Batched: prefix dims must match, last two are (M,K) and (K,N)
        assert 2 < len(shape_1) == len(shape_2) and shape_1[:-2] == shape_2[:-2], \
            f"MatMulIntegerOp: prefix mismatch {shape_1[:-2]} vs {shape_2[:-2]}"
        rows = shape_1[-2]
        cols = shape_1[-1]
        nrows = shape_2[-2]
        ncols = shape_2[-1]
        assert cols == nrows, f"MatMulIntegerOp: inner dims mismatch {cols} vs {nrows}"
        Bn = int(np.prod(shape_1[:-2]))
        # Optionally pre-subtract zero-points, then run pure batched matmul
        a_adj = tensor_in_1
        b_adj = tensor_in_2
        if provided_azp:
            mode_a = 2 if azp_size == Bn * rows else (1 if azp_size == rows else 0)
            a_adj = self.manager.tensor_t(np.zeros(Bn * rows * cols, dtype=np.int32))
            updated_tensors.append(a_adj)
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in_1, azp_tensor, a_adj],
                self.compiled_shader_sub_a,
                (rows, cols, Bn),
                [rows, cols, Bn, mode_a],
                []
            ))
        if provided_bzp:
            mode_b = 2 if bzp_size == Bn * ncols else (1 if bzp_size == ncols else 0)
            b_adj = self.manager.tensor_t(np.zeros(Bn * cols * ncols, dtype=np.int32))
            updated_tensors.append(b_adj)
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in_2, bzp_tensor, b_adj],
                self.compiled_shader_sub_b,
                (cols, ncols, Bn),
                [cols, ncols, Bn, mode_b],
                []
            ))

        tensor_out = self.manager.tensor_t(np.zeros(Bn * rows * ncols, dtype=np.int32))
        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm(
            [a_adj, b_adj, tensor_out],
            self.compiled_shader_matmul_batched,
            (rows, ncols, Bn),
            [rows, cols, ncols, Bn],
            []
        ))
        output_shape = shape_1[:-1] + [ncols]
        return [(tensor_out, output_shape)]
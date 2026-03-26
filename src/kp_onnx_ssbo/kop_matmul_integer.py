import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class MatMulIntegerOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader_matmul_2d = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly buffer buf_a {{ int in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly buffer buf_b {{ int in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_c {{ int C[]; }};
layout (std430, set = 0, binding = 3) readonly buffer UIParams {{ uint params[]; }};
void main() {{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    if (row >= M || col >= N) return;
    int acc = 0;
    uint a_base = row * K;
    uint b_base = col;
    for (uint k = 0; k < K; ++k, ++a_base, b_base += N) {{
        acc += in_tensor_1[a_base] * in_tensor_2[b_base];
    }}
    C[row * N + col] = acc;
}}
""")
        self.compiled_shader_matmul_batched = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly buffer buf_a {{ int in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly buffer buf_b {{ int in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_c {{ int C[]; }};
layout (std430, set = 0, binding = 3) readonly buffer UIParams {{ uint params[]; }};
void main() {{
    uint M = params[0];
    uint N = params[1];
    uint Bn = params[2];
    uint K = params[3];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint bid = gl_GlobalInvocationID.z;
    if (row >= M || col >= N || bid >= Bn) return;
    int acc = 0;
    uint a_base = bid * M * K + row * K;
    uint b_base = bid * K * N + col;
    for (uint k = 0; k < K; ++k, ++a_base, b_base += N) {{
        acc += in_tensor_1[a_base] * in_tensor_2[b_base];
    }}
    C[bid * M * N + row * N + col] = acc;
}}
""")
        # Zero-point preprocessing shaders
        self.compiled_shader_sub_a = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly buffer buf_in  {{ int in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly buffer buf_zp  {{ int zp[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_out {{ int out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly buffer UIParams {{ uint params[]; }};
void main(){{
    uint rows = params[0];
    uint cols = params[1];
    uint batches = params[2];
    uint mode = params[3];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint bid = gl_GlobalInvocationID.z;
    if (row >= rows || col >= cols || bid >= batches) return;
    uint p = (bid * rows + row) * cols + col;
    int v = in_tensor[p];
    int zpv = mode == 1 ? zp[row] : mode == 2 ? zp[bid * rows + row] : zp[0];
    v -= zpv;
    out_tensor[p] = v;
}}
""")
        self.compiled_shader_sub_b = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly buffer buf_in  {{ int in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly buffer buf_zp  {{ int zp[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_out {{ int out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly buffer UIParams {{ uint params[]; }};
void main(){{
    uint k_size = params[0];
    uint num_cols = params[1];
    uint batches = params[2];
    uint mode = params[3];
    uint k_index = gl_GlobalInvocationID.x;
    uint col_index = gl_GlobalInvocationID.y;
    uint batch_index = gl_GlobalInvocationID.z;
    if (k_index >= k_size || col_index >= num_cols || batch_index >= batches) return;
    uint p = (batch_index * k_size + k_index) * num_cols + col_index;
    int v = in_tensor[p];
    int zpv = mode == 1 ? zp[col_index] : mode == 2 ? zp[batch_index * num_cols + col_index] : zp[0];
    v -= zpv;
    out_tensor[p] = v;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulIntegerOp({device_name})"

    __str__ = __repr__

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

            a_adj = tensor_in_1
            b_adj = tensor_in_2
            if provided_azp:
                mode_a = 1 if azp_size == rows else 0
                a_adj = self.manager.tensor_t(np.zeros(rows * cols, dtype=np.int32))
                updated_tensors.append(a_adj)
                params = np.array([rows, cols, 1, mode_a], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
                wg_a = ((rows + LOCAL_X_3D - 1) // LOCAL_X_3D, (cols + LOCAL_Y_3D - 1) // LOCAL_Y_3D, 1)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in_1, azp_tensor, a_adj, param_in],
                    self.compiled_shader_sub_a,
                    wg_a,
                ))
            if provided_bzp:
                mode_b = 1 if bzp_size == ncols else 0
                b_adj = self.manager.tensor_t(np.zeros(cols * ncols, dtype=np.int32))
                updated_tensors.append(b_adj)
                params = np.array([cols, ncols, 1, mode_b], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
                wg_b = ((cols + LOCAL_X_3D - 1) // LOCAL_X_3D, (ncols + LOCAL_Y_3D - 1) // LOCAL_Y_3D, 1)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in_2, bzp_tensor, b_adj, param_in],
                    self.compiled_shader_sub_b,
                    wg_b,
                ))

            tensor_out = self.manager.tensor_t(np.zeros(rows * ncols, dtype=np.int32))
            updated_tensors.append(tensor_out)
            params = np.array([rows, ncols, cols], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            workgroup = ((rows + LOCAL_X_2D - 1) // LOCAL_X_2D, (ncols + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)
            updated_algorithms.append(self.manager.algorithm(
                [a_adj, b_adj, tensor_out, param_in],
                self.compiled_shader_matmul_2d,
                workgroup,
            ))
            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]

        assert 2 < len(shape_1) == len(shape_2) and shape_1[:-2] == shape_2[:-2], \
            f"MatMulIntegerOp: prefix mismatch {shape_1[:-2]} vs {shape_2[:-2]}"
        rows = shape_1[-2]
        cols = shape_1[-1]
        nrows = shape_2[-2]
        ncols = shape_2[-1]
        assert cols == nrows, f"MatMulIntegerOp: inner dims mismatch {cols} vs {nrows}"
        Bn = int(np.prod(shape_1[:-2]))

        a_adj = tensor_in_1
        b_adj = tensor_in_2
        if provided_azp:
            mode_a = 2 if azp_size == Bn * rows else (1 if azp_size == rows else 0)
            a_adj = self.manager.tensor_t(np.zeros(Bn * rows * cols, dtype=np.int32))
            updated_tensors.append(a_adj)
            params = np.array([rows, cols, Bn, mode_a], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            wg_a = ((rows + LOCAL_X_3D - 1) // LOCAL_X_3D, (cols + LOCAL_Y_3D - 1) // LOCAL_Y_3D, (Bn + LOCAL_Z_3D - 1) // LOCAL_Z_3D)
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in_1, azp_tensor, a_adj, param_in],
                self.compiled_shader_sub_a,
                wg_a,
            ))
        if provided_bzp:
            mode_b = 2 if bzp_size == Bn * ncols else (1 if bzp_size == ncols else 0)
            b_adj = self.manager.tensor_t(np.zeros(Bn * cols * ncols, dtype=np.int32))
            updated_tensors.append(b_adj)
            params = np.array([cols, ncols, Bn, mode_b], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            wg_b = ((cols + LOCAL_X_3D - 1) // LOCAL_X_3D, (ncols + LOCAL_Y_3D - 1) // LOCAL_Y_3D, (Bn + LOCAL_Z_3D - 1) // LOCAL_Z_3D)
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in_2, bzp_tensor, b_adj, param_in],
                self.compiled_shader_sub_b,
                wg_b,
            ))

        tensor_out = self.manager.tensor_t(np.zeros(Bn * rows * ncols, dtype=np.int32))
        updated_tensors.append(tensor_out)
        params = np.array([rows, ncols, Bn, cols], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        workgroup = ((rows + LOCAL_X_3D - 1) // LOCAL_X_3D, (ncols + LOCAL_Y_3D - 1) // LOCAL_Y_3D, (Bn + LOCAL_Z_3D - 1) // LOCAL_Z_3D)
        updated_algorithms.append(self.manager.algorithm(
            [a_adj, b_adj, tensor_out, param_in],
            self.compiled_shader_matmul_batched,
            workgroup,
        ))
        output_shape = shape_1[:-1] + [ncols]
        return [(tensor_out, output_shape)]

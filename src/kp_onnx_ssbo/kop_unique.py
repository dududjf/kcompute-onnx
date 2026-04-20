import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_1D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class UniqueOp:
    def __init__(self, manager: kp.Manager, axis=None, sorted=1):
        self.manager = manager
        self.axis = axis
        self.sorted = sorted  # Only sorted=1 is supported

        # Shader for Bitonic Sort (sort indices by lexicographic order)
        # workgroup = (n_pow2, 1, 1), local_size_x = LOCAL_X_1D
        self.compiled_shader_sort_indices = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf     {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1)           buffer OutBuf    {{ uint out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams  {{ uint params[]; }};

bool lex_less(uint su, uint sv, uint leading, uint trailing, uint stride)
{{
    uint off_u = su * trailing;
    uint off_v = sv * trailing;
    uint jump = stride - trailing;

    for (uint ld = 0; ld < leading; ++ld) {{
        for (uint t = 0; t < trailing; ++t, ++off_u, ++off_v) {{
            float a = in_tensor[off_u];
            float b = in_tensor[off_v];
            if (a < b) return true;
            if (a > b) return false;
        }}
        off_u += jump;
        off_v += jump;
    }}
    return false;
}}

void main()
{{
    uint tid = gl_GlobalInvocationID.x;

    uint size = params[0];
    uint leading = params[1];
    uint trailing = params[2];
    uint stride = params[3];
    uint k = params[4];
    uint j = params[5];

    if (tid >= size) return;
    uint ixj = tid ^ j;

    if (ixj > tid && ixj < size) {{
        uint u = out_tensor[tid];
        uint v = out_tensor[ixj];
        bool should_swap = lex_less(v, u, leading, trailing, stride);

        if ((tid & k) == 0) {{
            if (should_swap) {{
                out_tensor[tid] = v;
                out_tensor[ixj] = u;
            }}
        }} else {{
            if (!should_swap) {{
                out_tensor[tid] = v;
                out_tensor[ixj] = u;
            }}
        }}
    }}
}}
""")

        # Compare adjacent slices, mark heads using atomicOr
        # workgroup = ((size - 1 + LOCAL_X_3D - 1) // LOCAL_X_3D, (leading + LOCAL_Y_3D - 1) // LOCAL_Y_3D, (trailing + LOCAL_Z_3D - 1) // LOCAL_Z_3D)
        self.compiled_shader_compare_adjacent = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf       {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer IdxBuf      {{ uint index_sorted[]; }};
layout (std430, set = 0, binding = 2)           buffer OutBuf      {{ uint is_head[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams    {{ uint params[]; }};

void main()
{{
    uint u = gl_GlobalInvocationID.x + 1;  // Start from 1, skip first slice
    uint ld = gl_GlobalInvocationID.y;
    uint tt = gl_GlobalInvocationID.z;

    uint size = params[0];
    uint leading = params[1];
    uint trailing = params[2];
    uint stride = size * trailing;

    if (u >= size || ld >= leading || tt >= trailing) return;

    uint cur_idx = index_sorted[u];
    uint prev_idx = index_sorted[u - 1];

    uint offset_cur = ld * stride + cur_idx * trailing + tt;
    uint offset_prev = ld * stride + prev_idx * trailing + tt;

    if (in_tensor[offset_cur] != in_tensor[offset_prev]) {{
        atomicOr(is_head[u], 1u);
    }}
}}
""")

        # Detect runs based on is_head array (single-threaded)
        # workgroup = (1, 1, 1)
        self.compiled_shader_detect_runs = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (std430, set = 0, binding = 0) readonly  buffer IdxBuf     { uint index_sorted[]; };
layout (std430, set = 0, binding = 1) readonly  buffer HeadBuf    { uint is_head[]; };
layout (std430, set = 0, binding = 2)           buffer IndBuf     { uint indices_out[]; };
layout (std430, set = 0, binding = 3)           buffer CntBuf     { uint counts_out[]; };
layout (std430, set = 0, binding = 4) writeonly buffer NumBuf     { uint num_runs_buf[]; };
layout (std430, set = 0, binding = 5) readonly  buffer UIParams   { uint params[]; };

void main()
{
    uint size = params[0];

    uint num_runs = 0;
    for (uint u = 0; u < size; ++u) {
        uint cur_idx = index_sorted[u];
        bool is_head_flag = (is_head[u] != 0);

        if (is_head_flag) {
            indices_out[num_runs] = cur_idx;
            counts_out[num_runs] = 1;
            num_runs++;
        } else {
            counts_out[num_runs - 1] += 1;
            if (cur_idx < indices_out[num_runs - 1]) {
                indices_out[num_runs - 1] = cur_idx;
            }
        }
    }

    num_runs_buf[0] = num_runs;
}
""")

        # Copy unique values in sorted order
        # workgroup = (1, (trailing + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)
        self.compiled_shader_copy_unique = compile_source(f"""
#version 450
layout (local_size_x = 1, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf      {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer IdxBuf     {{ uint index_sorted[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer CntBuf     {{ uint counts_out[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer NumBuf     {{ uint num_runs_buf[]; }};
layout (std430, set = 0, binding = 4) writeonly buffer OutBuf     {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 5) writeonly buffer InvBuf     {{ uint inverse_out[]; }};
layout (std430, set = 0, binding = 6) readonly  buffer UIParams   {{ uint params[]; }};

void copy_slice_tr(uint s_from, uint s_to, uint tr, uint stride, uint leading, uint trailing)
{{
    uint src = s_from * trailing + tr;
    uint dst = s_to * trailing + tr;
    for (uint ld = 0; ld < leading; ++ld, src += stride, dst += stride) {{
        out_tensor[dst] = in_tensor[src];
    }}
}}

void main()
{{
    uint tr = gl_GlobalInvocationID.y;

    uint size = params[0];
    uint leading = params[1];
    uint trailing = params[2];
    uint stride = size * trailing;

    if (tr >= trailing) return;

    uint num_runs = num_runs_buf[0];

    uint start = 0;
    for (uint r = 0; r < num_runs; ++r) {{
        uint run_len = counts_out[r];

        copy_slice_tr(index_sorted[start], r, tr, stride, leading, trailing);

        if (tr == 0) {{
            for (uint k = 0; k < run_len; ++k) {{
                uint orig = index_sorted[start + k];
                inverse_out[orig] = r;
            }}
        }}
        start += run_len;
    }}
}}
""")

    def __repr__(self):
        return f"UniqueOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([t[0] for t in output_tensor_and_shape]))
        seq.eval()

        output_list = []
        for tensor, output_shape in output_tensor_and_shape:
            output = tensor.data().reshape(output_shape)
            output_list.append(output)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return output_list

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert self.sorted == 1, "Only sorted=1 is supported in this UniqueOp implementation."
        tensor_in, shape_in = input_tensors[0]

        if self.axis is not None:
            axis = self.axis + len(shape_in) if self.axis < 0 else self.axis
            size = shape_in[axis]
            leading = int(np.prod(shape_in[:axis])) if axis > 0 else 1
            trailing = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1
        else:
            # axis=None: flatten the array, treat as 1D
            size = int(np.prod(shape_in))
            leading = 1
            trailing = 1
            shape_in = [size]

        # Step 1: Sort indices using Bitonic Sort O(n log^2 n)
        tensor_index_sorted = self.manager.tensor_t(np.arange(size, dtype=np.uint32))
        updated_tensors.append(tensor_index_sorted)

        # Find next power of 2 >= size for proper bitonic sort
        n_pow2 = 1
        while n_pow2 < size:
            n_pow2 *= 2

        stride = size * trailing

        # Create params buffer for sort shader
        k = 2
        while k <= n_pow2:
            j = k >> 1
            while j > 0:
                sort_params = np.array([size, leading, trailing, stride, k, j], dtype=np.uint32)
                sort_param_buf = self.manager.tensor_t(sort_params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([sort_param_buf])).eval()

                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_index_sorted, sort_param_buf],
                    self.compiled_shader_sort_indices,
                    ((n_pow2 + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1),
                ))
                j >>= 1
            k <<= 1

        tensor_unique_out = self.manager.tensor(np.zeros(int(np.prod(shape_in)), dtype=np.float32))
        tensor_indices_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_counts_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_inverse_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_num_runs = self.manager.tensor_t(np.zeros(1, dtype=np.uint32))
        is_head_init = np.zeros(size, dtype=np.uint32)
        is_head_init[0] = 1  # First slice is always a head
        tensor_is_head = self.manager.tensor_t(is_head_init)
        updated_tensors.extend([tensor_unique_out, tensor_indices_out, tensor_counts_out,
                                tensor_inverse_out, tensor_num_runs, tensor_is_head])

        # Step 2a: Compare adjacent slices in parallel
        compare_params = np.array([size, leading, trailing], dtype=np.uint32)
        compare_param_buf = self.manager.tensor_t(compare_params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([compare_param_buf])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_index_sorted, tensor_is_head, compare_param_buf],
            self.compiled_shader_compare_adjacent,
            ((size - 1 + LOCAL_X_3D - 1) // LOCAL_X_3D,
             (leading + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
             (trailing + LOCAL_Z_3D - 1) // LOCAL_Z_3D),
        ))

        # Step 2b: Detect runs based on is_head array (single-threaded)
        detect_params = np.array([size], dtype=np.uint32)
        detect_param_buf = self.manager.tensor_t(detect_params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([detect_param_buf])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_index_sorted, tensor_is_head, tensor_indices_out, tensor_counts_out,
             tensor_num_runs, detect_param_buf],
            self.compiled_shader_detect_runs,
            (1, 1, 1),
        ))

        # Step 3: Copy unique values (parallel across trailing dimension)
        copy_params = np.array([size, leading, trailing], dtype=np.uint32)
        copy_param_buf = self.manager.tensor_t(copy_params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([copy_param_buf])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_index_sorted, tensor_counts_out, tensor_num_runs,
             tensor_unique_out, tensor_inverse_out, copy_param_buf],
            self.compiled_shader_copy_unique,
            (1, (trailing + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1),
        ))

        return [(tensor_unique_out, shape_in), (tensor_indices_out, [size]),
                (tensor_inverse_out, [size]), (tensor_counts_out, [size])]

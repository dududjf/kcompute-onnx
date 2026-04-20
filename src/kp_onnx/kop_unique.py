import numpy as np
import kp
from .shader_utils import compile_source


class UniqueOp:
    def __init__(self, manager: kp.Manager, axis=None, sorted=1):
        self.manager = manager
        self.axis = axis
        self.sorted = sorted  # Only sorted=1 is supported
        
        # Shader for Bitonic Sort (sort indices by lexicographic order)
        self.compiled_shader_sort_indices = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_input { float tensor_in[]; };
layout (binding = 1) buffer buf_idx_sorted { uint tensor_index_sorted[]; };

layout (constant_id = 0) const float size_f = 0;
layout (constant_id = 1) const float leading_f = 0;
layout (constant_id = 2) const float trailing_f = 0;
layout (constant_id = 3) const float stride_f = 0;
layout (constant_id = 4) const float k_f = 0;
layout (constant_id = 5) const float j_f = 0;

bool lex_less(uint su, uint sv, uint leading, uint trailing, uint stride)
{
    uint off_u = su * trailing;
    uint off_v = sv * trailing;
    uint jump = stride - trailing;  // 每完成一个 trailing 块后的跳跃量
    
    // 每个切片包含 leading个块, 每个块包含 trailing个连续元素
    for (uint ld = 0; ld < leading; ++ld) {
        for (uint t = 0; t < trailing; ++t, ++off_u, ++off_v) {
            float a = tensor_in[off_u];
            float b = tensor_in[off_v];
            if (a < b) return true;
            if (a > b) return false;
        }
        off_u += jump;
        off_v += jump;
    }
    return false;
}

void main()
{
    uint tid = gl_GlobalInvocationID.x;
    uint size = uint(size_f);
    uint leading = uint(leading_f);
    uint trailing = uint(trailing_f);
    uint stride = uint(stride_f);
    uint k = uint(k_f);
    uint j = uint(j_f);
    
    if (tid >= size) return;
    uint ixj = tid ^ j;
    
    if (ixj > tid && ixj < size) {
        uint u = tensor_index_sorted[tid];
        uint v = tensor_index_sorted[ixj];
        bool should_swap = lex_less(v, u, leading, trailing, stride);
        
        if ((tid & k) == 0) {
            // 当前区间为升序
            if (should_swap) {
                tensor_index_sorted[tid] = v;
                tensor_index_sorted[ixj] = u;
            }
        } else {
            // 当前区间为降序
            if (!should_swap) {
                tensor_index_sorted[tid] = v;
                tensor_index_sorted[ixj] = u;
            }
        }
    }
}
""")

        # 在排序后的索引数组中，并行比较相邻 slice，标记哪些位置是 head
        # workgroup = (size, leading, trailing)
        self.compiled_shader_compare_adjacent = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_input { float input_buf[]; };
layout (binding = 1) readonly buffer buf_index_sorted { uint index_sorted[]; };
layout (binding = 2) buffer buf_is_head { uint is_head[]; };

layout (constant_id = 0) const float size_f = 0;
layout (constant_id = 1) const float leading_f = 0;
layout (constant_id = 2) const float trailing_f = 0;

void main()
{
    uint u = gl_GlobalInvocationID.x + 1;  // 从 1 开始，跳过第一个 slice
    uint ld = gl_GlobalInvocationID.y;
    uint tt = gl_GlobalInvocationID.z;
    
    uint size = uint(size_f);
    uint leading = uint(leading_f);
    uint trailing = uint(trailing_f);
    uint stride = size * trailing;
    
    uint cur_idx = index_sorted[u];
    uint prev_idx = index_sorted[u - 1];
    
    uint offset_cur = ld * stride + cur_idx * trailing + tt;
    uint offset_prev = ld * stride + prev_idx * trailing + tt;
    
    if (input_buf[offset_cur] != input_buf[offset_prev]) {
        atomicOr(is_head[u], 1);
    }
}
""")

        # 根据 is_head 数组，统计 runs
        self.compiled_shader_detect_runs = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_index_sorted { uint index_sorted[]; };
layout (binding = 1) readonly buffer buf_is_head { uint is_head[]; };
layout (binding = 2) buffer buf_indices_out { uint indices_out[]; };
layout (binding = 3) buffer buf_counts_out { uint counts_out[]; };
layout (binding = 4) writeonly buffer buf_num_runs { uint num_runs_buf[]; };

layout (constant_id = 0) const float size_f = 0;

void main()
{
    uint size = uint(size_f);
    
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

        # Shader for Stage 1: Copy unique values in sorted order
        self.compiled_shader_copy_unique = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Inputs (readonly)
layout (binding = 0) readonly buffer buf_input { float input_buf[]; };
layout (binding = 1) readonly buffer buf_index_sorted { uint index_sorted[]; };
layout (binding = 2) readonly buffer buf_counts_out { uint counts_out[]; };
layout (binding = 3) readonly buffer buf_num_runs { uint num_runs_buf[]; };

// Outputs (writeonly)
layout (binding = 4) writeonly buffer buf_unique_out { float unique_out[]; };
layout (binding = 5) writeonly buffer buf_inverse_out { uint inverse_out[]; };

layout (constant_id = 0) const float size_f = 0;
layout (constant_id = 1) const float leading_f = 0;
layout (constant_id = 2) const float trailing_f = 0;

void copy_slice_tr(uint s_from, uint s_to, uint tr, uint stride, uint leading, uint trailing)
{
    uint src = s_from * trailing + tr;
    uint dst = s_to * trailing + tr;
    for (uint ld = 0; ld < leading; ++ld, src += stride, dst += stride) {
        unique_out[dst] = input_buf[src];
    }
}

void main()
{
    uint tr = gl_GlobalInvocationID.y;
    
    uint size = uint(size_f);
    uint leading = uint(leading_f);
    uint trailing = uint(trailing_f);
    uint stride = size * trailing;
    
    uint num_runs = num_runs_buf[0];
    
    uint start = 0;
    for (uint r = 0; r < num_runs; ++r) {
        uint run_len = counts_out[r];
        
        copy_slice_tr(index_sorted[start], r, tr, stride, leading, trailing);
        
        if (tr == 0) {
            for (uint k = 0; k < run_len; ++k) {
                uint orig = index_sorted[start + k];
                inverse_out[orig] = r;
            }
        }
        start += run_len;
    }
}
""")

    def __repr__(self):
        return f"UniqueOp({self.manager.get_device_properties()['device_name']})"

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
            trailing = int(np.prod(shape_in[axis+1:])) if axis + 1 < len(shape_in) else 1
        else:
            # axis=None: flatten the array, treat as 1D
            size = int(np.prod(shape_in))
            leading = 1
            trailing = 1
            shape_in = [size]  # Override shape for 1D processing

        # Step 1: Sort indices using Bitonic Sort O(n log^2 n)
        tensor_index_sorted = self.manager.tensor_t(np.arange(size, dtype=np.uint32))
        updated_tensors.append(tensor_index_sorted)
        
        # Find next power of 2 >= size for proper bitonic sort
        n_pow2 = 1
        while n_pow2 < size:
            n_pow2 *= 2
        
        stride = size * trailing
        k = 2
        while k <= n_pow2:
            j = k >> 1
            while j > 0:
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_index_sorted],
                    self.compiled_shader_sort_indices,
                    (n_pow2, 1, 1),
                    [size, leading, trailing, stride, k, j],
                    []
                ))
                j >>= 1
            k <<= 1

        tensor_unique_out = self.manager.tensor(np.zeros(int(np.prod(shape_in)), dtype=np.float32))
        tensor_indices_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_counts_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_inverse_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_num_runs = self.manager.tensor_t(np.zeros(1, dtype=np.uint32))
        is_head_init = np.zeros(size, dtype=np.uint32)
        is_head_init[0] = 1  # 第一个 slice 永远是 head
        tensor_is_head = self.manager.tensor_t(is_head_init)
        updated_tensors.extend([tensor_unique_out, tensor_indices_out, tensor_counts_out, tensor_inverse_out, tensor_num_runs, tensor_is_head])
        
        # Step 2a: Compare adjacent slices in parallel, workgroup = (size - 1, leading, trailing)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_index_sorted, tensor_is_head],
            self.compiled_shader_compare_adjacent,
            (size - 1, leading, trailing),
            [size, leading, trailing],
            []
        ))
        
        # Step 2b: Detect runs based on is_head array (single-threaded)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_index_sorted, tensor_is_head, tensor_indices_out, tensor_counts_out, tensor_num_runs],
            self.compiled_shader_detect_runs,
            (1, 1, 1),
            [size],
            []
        ))
        
        # Step 3: Copy unique values (parallel across trailing dimension)
        # Tensor order: inputs first, then outputs
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_index_sorted, tensor_counts_out, tensor_num_runs,
             tensor_unique_out, tensor_inverse_out],
            self.compiled_shader_copy_unique,
            (1, trailing, 1),
            [size, leading, trailing],
            []
        ))

        return [(tensor_unique_out, shape_in), (tensor_indices_out, [size]),
                (tensor_inverse_out, [size]), (tensor_counts_out, [size])]

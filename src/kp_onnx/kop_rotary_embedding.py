import numpy as np
import kp
from .shader_utils import compile_source


class RotaryEmbeddingOp:
    def __init__(self, manager: kp.Manager, interleaved=None, rotary_embedding_dim=None, num_heads=None):
        self.interleaved = interleaved if interleaved is not None else 0
        self.rotary_embedding_dim = rotary_embedding_dim
        self.num_heads = num_heads
        self.manager = manager
        # 切片 Shader
        self.shader_slice = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) writeonly buffer out_buf    { float out_tensor[];    };

layout(constant_id = 0) const float elements_per_slice_f = 0;
layout(constant_id = 1) const float start_f = 0;
layout(constant_id = 2) const float size_f = 0;
layout(constant_id = 3) const float axis_size_f = 0;
layout(constant_id = 4) const float step_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint elements_per_slice = uint(elements_per_slice_f);
    uint start = uint(start_f);
    uint size = uint(size_f);
    uint axis_size = uint(axis_size_f);
    uint step = uint(step_f);

    uint in_offset = (gx * axis_size + start + gy * step) * elements_per_slice + gz;
    uint out_offset = (gx * size + gy) * elements_per_slice + gz;

    out_tensor[out_offset] = in_tensor[in_offset];
}
""")
        # 旋转嵌入计算的 Shader - 交错模式
        self.shader_rotary_interleaved = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InputBuf { float input_data[]; };
layout(set=0, binding=1) readonly buffer CosBuf { float cos_data[]; };
layout(set=0, binding=2) readonly buffer SinBuf { float sin_data[]; };
layout(set=0, binding=3) writeonly buffer OutputBuf { float output_data[]; };

layout(constant_id = 0) const float batch_f = 0;
layout(constant_id = 1) const float seq_len_f = 0;
layout(constant_id = 2) const float num_heads_f = 0;
layout(constant_id = 3) const float head_size_f = 0;
layout(constant_id = 4) const float rotary_dim_f = 0;
layout(constant_id = 5) const float rotary_half_f = 0;
layout(constant_id = 6) const float input_seq_head_f = 0;
layout(constant_id = 7) const float input_head_f = 0;
layout(constant_id = 8) const float cos_sin_seq_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint s = gl_GlobalInvocationID.y;
    uint h = gl_GlobalInvocationID.z;

    uint batch = uint(batch_f);
    uint seq_len = uint(seq_len_f);
    uint num_heads = uint(num_heads_f);
    uint head_size = uint(head_size_f);
    uint rotary_dim = uint(rotary_dim_f);
    uint rotary_half = uint(rotary_half_f);
    uint input_seq_head = uint(input_seq_head_f);
    uint input_head = uint(input_head_f);
    uint cos_sin_seq = uint(cos_sin_seq_f);

    // 计算基址
    uint input_base = b * input_seq_head + s * input_head + h * head_size;
    uint cos_sin_base = b * cos_sin_seq + s * rotary_half;

    // 交错模式：在 0/1, 2/3, ... 上旋转
    for (uint i = 0; i < rotary_half; ++i) {
        float c = cos_data[cos_sin_base + i];
        float sn = sin_data[cos_sin_base + i];

        uint idx1 = input_base + (i << 1u);
        uint idx2 = idx1 + 1u;
        
        float x1 = input_data[idx1];
        float x2 = input_data[idx2];

        output_data[idx1] = c * x1 - sn * x2;
        output_data[idx2] = sn * x1 + c * x2;
    }
    // 复制未旋转部分
    for (uint i = rotary_dim; i < head_size; ++i) {
        output_data[input_base + i] = input_data[input_base + i];
    }
}
""")
        # 旋转嵌入计算的 Shader - 非交错模式
        self.shader_rotary_non_interleaved = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InputBuf { float input_data[]; };
layout(set=0, binding=1) readonly buffer CosBuf { float cos_data[]; };
layout(set=0, binding=2) readonly buffer SinBuf { float sin_data[]; };
layout(set=0, binding=3) writeonly buffer OutputBuf { float output_data[]; };

layout(constant_id = 0) const float batch_f = 0;
layout(constant_id = 1) const float seq_len_f = 0;
layout(constant_id = 2) const float num_heads_f = 0;
layout(constant_id = 3) const float head_size_f = 0;
layout(constant_id = 4) const float rotary_dim_f = 0;
layout(constant_id = 5) const float rotary_half_f = 0;
layout(constant_id = 6) const float input_seq_head_f = 0;
layout(constant_id = 7) const float input_head_f = 0;
layout(constant_id = 8) const float cos_sin_seq_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint s = gl_GlobalInvocationID.y;
    uint h = gl_GlobalInvocationID.z;

    uint batch = uint(batch_f);
    uint seq_len = uint(seq_len_f);
    uint num_heads = uint(num_heads_f);
    uint head_size = uint(head_size_f);
    uint rotary_dim = uint(rotary_dim_f);
    uint rotary_half = uint(rotary_half_f);
    uint input_seq_head = uint(input_seq_head_f);
    uint input_head = uint(input_head_f);
    uint cos_sin_seq = uint(cos_sin_seq_f);

    // 计算基址
    uint input_base = b * input_seq_head + s * input_head + h * head_size;
    uint cos_sin_base = b * cos_sin_seq + s * rotary_half;

    // 非交错模式：分别旋转前后两个 half
    for (uint i = 0; i < rotary_half; ++i) {
        float c = cos_data[cos_sin_base + i];
        float sn = sin_data[cos_sin_base + i];

        uint idx1 = input_base + i;
        uint idx2 = input_base + rotary_half + i;
        
        float x1 = input_data[idx1];
        float x2 = input_data[idx2];

        output_data[idx1] = c * x1 - sn * x2;
        output_data[idx2] = sn * x1 + c * x2;
    }
    // 复制未旋转部分
    for (uint i = rotary_dim; i < head_size; ++i) {
        output_data[input_base + i] = input_data[input_base + i];
    }
}
""")
        # 3D 转 4D 变换 Shader
        self.shader_reshape_3d_to_4d = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InputBuf { float input_data[]; };
layout(set=0, binding=1) writeonly buffer OutputBuf { float output_data[]; };

layout(constant_id = 0) const float batch_f = 0;
layout(constant_id = 1) const float seq_len_f = 0;
layout(constant_id = 2) const float num_heads_f = 0;
layout(constant_id = 3) const float head_size_f = 0;
layout(constant_id = 4) const float hidden_size_f = 0;
layout(constant_id = 5) const float seq_hidden_f = 0;
layout(constant_id = 6) const float seq_num_heads_head_f = 0;
layout(constant_id = 7) const float num_heads_head_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint s = gl_GlobalInvocationID.y;
    uint h = gl_GlobalInvocationID.z;

    uint batch = uint(batch_f);
    uint seq_len = uint(seq_len_f);
    uint num_heads = uint(num_heads_f);
    uint head_size = uint(head_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint seq_hidden = uint(seq_hidden_f);
    uint seq_num_heads_head = uint(seq_num_heads_head_f);
    uint num_heads_head = uint(num_heads_head_f);

    // Input: (batch, seq_len, hidden_size)
    uint in_base = b * seq_hidden + s * hidden_size + h * head_size;

    // Output: (batch, seq_len, num_heads, head_size)
    uint out_base = b * seq_num_heads_head + s * num_heads_head + h * head_size;

    for (uint i = 0; i < head_size; ++i) {
        output_data[out_base + i] = input_data[in_base + i];
    }
}
""")
        # 4D 到 3D 还原 Shader
        self.shader_reshape_4d_to_3d = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InputBuf { float input_data[]; };
layout(set=0, binding=1) writeonly buffer OutputBuf { float output_data[]; };

layout(constant_id = 0) const float batch_f = 0;
layout(constant_id = 1) const float seq_len_f = 0;
layout(constant_id = 2) const float num_heads_f = 0;
layout(constant_id = 3) const float head_size_f = 0;
layout(constant_id = 4) const float hidden_size_f = 0;
layout(constant_id = 5) const float seq_hidden_f = 0;
layout(constant_id = 6) const float seq_num_heads_head_f = 0;
layout(constant_id = 7) const float num_heads_head_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint s = gl_GlobalInvocationID.y;
    uint h = gl_GlobalInvocationID.z;

    uint batch = uint(batch_f);
    uint seq_len = uint(seq_len_f);
    uint num_heads = uint(num_heads_f);
    uint head_size = uint(head_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint seq_hidden = uint(seq_hidden_f);
    uint seq_num_heads_head = uint(seq_num_heads_head_f);
    uint num_heads_head = uint(num_heads_head_f);

    // Input: (batch, seq_len, num_heads, head_size)
    uint in_base = b * seq_num_heads_head + s * num_heads_head + h * head_size;

    // Output: (batch, seq_len, hidden_size)
    uint out_base = b * seq_hidden + s * hidden_size + h * head_size;

    for (uint i = 0; i < head_size; ++i) {
        output_data[out_base + i] = input_data[in_base + i];
    }
}
""")
        # 4D transpose Shader
        self.shader_transpose_4d = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InputBuf { float input_data[]; };
layout(set=0, binding=1) writeonly buffer OutputBuf { float output_data[]; };

layout(constant_id = 0) const float batch_f = 0;
layout(constant_id = 1) const float dim1_f = 0;
layout(constant_id = 2) const float dim2_f = 0;
layout(constant_id = 3) const float dim3_f = 0;
layout(constant_id = 4) const float d2_d3_f = 0;
layout(constant_id = 5) const float d1_d2_d3_f = 0;
layout(constant_id = 6) const float dim1_dim3_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint d1 = gl_GlobalInvocationID.y;
    uint d2 = gl_GlobalInvocationID.z;

    uint batch = uint(batch_f);
    uint dim1 = uint(dim1_f);
    uint dim2 = uint(dim2_f);
    uint dim3 = uint(dim3_f);
    uint d2_d3 = uint(d2_d3_f);
    uint d1_d2_d3 = uint(d1_d2_d3_f);
    uint dim1_dim3 = uint(dim1_dim3_f);

    // Input: (batch, dim1, dim2, dim3), Output: (batch, dim2, dim1, dim3)
    uint in_base = b * d1_d2_d3 + d1 * d2_d3 + d2 * dim3;
    uint out_base = b * d1_d2_d3 + d2 * dim1_dim3 + d1 * dim3;

    for (uint i = 0; i < dim3; ++i) {
        output_data[out_base + i] = input_data[in_base + i];
    }
}
""")
        # position_ids 收集 Shader
        self.shader_gather_position = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer CacheBuf { float cache_data[]; };
layout(set=0, binding=1) readonly buffer PositionIdsBuf { float position_ids_data[]; };
layout(set=0, binding=2) writeonly buffer OutputBuf { float output_data[]; };

layout(constant_id = 0) const float batch_f = 0;
layout(constant_id = 1) const float seq_len_f = 0;
layout(constant_id = 2) const float rotary_half_f = 0;
layout(constant_id = 3) const float seq_rotary_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint s = gl_GlobalInvocationID.y;
    uint i = gl_GlobalInvocationID.z;

    uint batch = uint(batch_f);
    uint seq_len = uint(seq_len_f);
    uint rotary_half = uint(rotary_half_f);
    uint seq_rotary = uint(seq_rotary_f);

    uint pos_id = uint(position_ids_data[b * seq_len + s]);
    uint cache_idx = pos_id * rotary_half + i;
    uint out_idx = b * seq_rotary + s * rotary_half + i;

    output_data[out_idx] = cache_data[cache_idx];
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"RotaryEmbeddingOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is None:
                input_tensors.append(None)
            else:
                numpy_in = inp.reshape(-1).astype(np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shapes = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        all_tensors_to_sync = [t[0] for t in input_tensors if t is not None] + updated_tensors
        seq.record(kp.OpTensorSyncDevice(all_tensors_to_sync))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([t for t, _ in output_tensor_and_shapes]))
        seq.eval()

        outputs = []
        for tensor_out, output_shape in output_tensor_and_shapes:
            outputs.append(tensor_out.data().reshape(output_shape))

        for t in input_tensors:
            if t is not None:
                tensor, _ = t
                del tensor
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        # 解析输入
        input_tensor, input_shape = input_tensors[0]
        cos_cache_tensor, cos_cache_shape = input_tensors[1]
        sin_cache_tensor, sin_cache_shape = input_tensors[2]
        position_ids = input_tensors[3] if len(input_tensors) > 3 and input_tensors[3] is not None else None

        input_shape_len = len(input_shape)

        # 第一步：将输入transpose为 4D（batch，seq_len，num_heads，head_size）
        if input_shape_len == 4:
            # 输入为（batch，num_heads，seq_len，head_size），需要transpose为（batch，seq_len，num_heads，head_size）
            batch = input_shape[0]
            num_heads = input_shape[1]
            seq_len = input_shape[2]
            head_size = input_shape[3]

            # transpose
            total_size = batch * seq_len * num_heads * head_size
            input_4d_tensor = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
            updated_tensors.append(input_4d_tensor)

            # 预计算常量
            d2_d3 = seq_len * head_size
            d1_d2_d3 = num_heads * d2_d3
            dim1_dim3 = num_heads * head_size

            updated_algorithms.append(self.manager.algorithm(
                [input_tensor, input_4d_tensor],
                self.shader_transpose_4d,
                (batch, num_heads, seq_len),
                [batch, num_heads, seq_len, head_size, d2_d3, d1_d2_d3, dim1_dim3],
                []
            ))

            input_tensor = input_4d_tensor
        else:
            # 输入为（batch，seq_len，hidden_size），需要reshape为（batch，seq_len，num_heads，head_size）
            batch = input_shape[0]
            seq_len = input_shape[1]
            hidden_size = input_shape[2]

            assert self.num_heads is not None, "num_heads must be provided for 3D input"
            num_heads = self.num_heads
            head_size = hidden_size // num_heads

            # reshape
            total_size = batch * seq_len * num_heads * head_size
            input_4d_tensor = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
            updated_tensors.append(input_4d_tensor)

            # 预计算常量
            seq_hidden = seq_len * hidden_size
            seq_num_heads_head = seq_len * num_heads * head_size
            num_heads_head = num_heads * head_size

            updated_algorithms.append(self.manager.algorithm(
                [input_tensor, input_4d_tensor],
                self.shader_reshape_3d_to_4d,
                (batch, seq_len, num_heads),
                [batch, seq_len, num_heads, head_size, hidden_size, seq_hidden, seq_num_heads_head, num_heads_head],
                []
            ))

            input_tensor = input_4d_tensor

        # 第二步：确定 rotary_embedding_dim
        if self.rotary_embedding_dim is None or self.rotary_embedding_dim == 0:
            rotary_dim = head_size
        else:
            rotary_dim = self.rotary_embedding_dim
        rotary_half = rotary_dim >> 1

        # 第 2.5 步：切片 cos 和 sin 缓存以匹配 rotary_dim
        def slice_cache(cache_tensor, pre_elements, cache_dim):
            """对单个 cache 张量进行切片的辅助函数"""
            elements_per_slice = 1
            total_elements = pre_elements * rotary_half

            cache_sliced = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
            updated_tensors.append(cache_sliced)

            workgroup = (pre_elements, rotary_half, elements_per_slice)
            updated_algorithms.append(self.manager.algorithm(
                [cache_tensor, cache_sliced],
                self.shader_slice,
                workgroup,
                [elements_per_slice, 0, rotary_half, cache_dim, 1],
                []
            ))

            return cache_sliced

        cos_cache_shape_len = len(cos_cache_shape)

        if cos_cache_shape_len == 2:
            # 形状：（max_seq_len，cache_dim）
            cache_dim = cos_cache_shape[1]
            if cache_dim > rotary_half:
                # 沿轴 1 切片 [0:rotary_half]
                max_seq_len = cos_cache_shape[0]
                pre_elements = max_seq_len

                cos_cache_tensor = slice_cache(cos_cache_tensor, pre_elements, cache_dim)
                sin_cache_tensor = slice_cache(sin_cache_tensor, pre_elements, cache_dim)

        elif cos_cache_shape_len == 3:
            # 形状：（batch，seq_len，cache_dim）
            cache_dim = cos_cache_shape[2]
            if cache_dim > rotary_half:
                # 沿轴 2 切片 [0:rotary_half]
                cache_batch = cos_cache_shape[0]
                cache_seq = cos_cache_shape[1]
                pre_elements = cache_batch * cache_seq

                cos_cache_tensor = slice_cache(cos_cache_tensor, pre_elements, cache_dim)
                sin_cache_tensor = slice_cache(sin_cache_tensor, pre_elements, cache_dim)


        # 第三步：如果提供了 position_ids，则处理 cos 和 sin 缓存
        if position_ids is not None:
            position_ids_tensor, position_ids_shape = position_ids

            # 收集 cos 缓存
            gather_size = batch * seq_len * rotary_half
            cos_gathered = self.manager.tensor(np.zeros(gather_size, dtype=np.float32))
            updated_tensors.append(cos_gathered)

            # 预计算常量
            seq_rotary = seq_len * rotary_half

            updated_algorithms.append(self.manager.algorithm(
                [cos_cache_tensor, position_ids_tensor, cos_gathered],
                self.shader_gather_position,
                (batch, seq_len, rotary_half),
                [batch, seq_len, rotary_half, seq_rotary],
                []
            ))

            # 收集 sin 缓存
            sin_gathered = self.manager.tensor(np.zeros(gather_size, dtype=np.float32))
            updated_tensors.append(sin_gathered)

            updated_algorithms.append(self.manager.algorithm(
                [sin_cache_tensor, position_ids_tensor, sin_gathered],
                self.shader_gather_position,
                (batch, seq_len, rotary_half),
                [batch, seq_len, rotary_half, seq_rotary],
                []
            ))

            cos_cache_tensor = cos_gathered
            sin_cache_tensor = sin_gathered

        # 第四步：应用旋转嵌入
        rotary_size = batch * seq_len * num_heads * head_size
        output_4d_tensor = self.manager.tensor(np.zeros(rotary_size, dtype=np.float32))
        updated_tensors.append(output_4d_tensor)

        # 预计算常量
        input_seq_head = seq_len * num_heads * head_size
        input_head = num_heads * head_size
        cos_sin_seq = seq_len * rotary_half

        # 根据 interleaved 值选择对应的 shader
        if self.interleaved == 1:
            shader_rotary = self.shader_rotary_interleaved
        else:
            shader_rotary = self.shader_rotary_non_interleaved

        updated_algorithms.append(self.manager.algorithm(
            [input_tensor, cos_cache_tensor, sin_cache_tensor, output_4d_tensor],
            shader_rotary,
            (batch, seq_len, num_heads),
            [batch, seq_len, num_heads, head_size, rotary_dim, rotary_half,
             input_seq_head, input_head, cos_sin_seq],
            []
        ))

        # 第五步：将输出transpose回原始形状
        if input_shape_len == 3:
            # 转换回 3D（batch，seq_len，hidden_size）
            hidden_size = num_heads * head_size
            output_size = batch * seq_len * hidden_size
            output_tensor = self.manager.tensor(np.zeros(output_size, dtype=np.float32))
            updated_tensors.append(output_tensor)

            # 预计算常量
            seq_hidden = seq_len * hidden_size
            seq_num_heads_head = seq_len * num_heads * head_size
            num_heads_head = num_heads * head_size

            updated_algorithms.append(self.manager.algorithm(
                [output_4d_tensor, output_tensor],
                self.shader_reshape_4d_to_3d,
                (batch, seq_len, num_heads),
                [batch, seq_len, num_heads, head_size, hidden_size, seq_hidden, seq_num_heads_head, num_heads_head],
                []
            ))

            output_shape = [batch, seq_len, hidden_size]
        else:
            # transpose回 4D（batch，num_heads，seq_len，head_size）
            output_size = batch * num_heads * seq_len * head_size
            output_tensor = self.manager.tensor(np.zeros(output_size, dtype=np.float32))
            updated_tensors.append(output_tensor)

            # 预计算常量
            # 工作组为（batch，seq_len，num_heads）， dim1=seq_len，dim2=num_heads，dim3=head_size
            d2_d3 = num_heads * head_size
            d1_d2_d3 = seq_len * num_heads * head_size
            dim1_dim3 = seq_len * head_size

            updated_algorithms.append(self.manager.algorithm(
                [output_4d_tensor, output_tensor],
                self.shader_transpose_4d,
                (batch, seq_len, num_heads),
                [batch, seq_len, num_heads, head_size, d2_d3, d1_d2_d3, dim1_dim3],
                []
            ))

            output_shape = [batch, num_heads, seq_len, head_size]

        return [(output_tensor, output_shape)]

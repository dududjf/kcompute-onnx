import numpy as np
import kp
from .shader_utils import compile_source


class AttentionOp:
    def __init__(self, manager: kp.Manager, scale=None, is_causal=0, q_num_heads=None, kv_num_heads=None,
                 softmax_precision=None, softcap=0.0, qk_matmul_output_mode=0):
        self.scale = scale
        self.is_causal = is_causal
        self.q_num_heads = q_num_heads
        self.kv_num_heads = kv_num_heads
        self.softmax_precision = softmax_precision
        self.softcap = softcap
        self.qk_matmul_output_mode = qk_matmul_output_mode
        self.manager = manager
        # Q*K^T计算的shader（批量矩阵乘法，K进行转置）
        self.shader_qk = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer QBuf { float q_data[]; };
layout(set=0, binding=1) readonly buffer KBuf { float k_data[]; };
layout(set=0, binding=2) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float head_size_f = 0;
layout(constant_id = 1) const float kv_seq_len_f = 0;
layout(constant_id = 2) const float scale_val_f = 0;
layout(constant_id = 3) const float q_seq_head_f = 0;
layout(constant_id = 4) const float q_batch_head_f = 0;
layout(constant_id = 5) const float k_seq_head_f = 0;
layout(constant_id = 6) const float k_batch_head_f = 0;
layout(constant_id = 7) const float out_seq_kv_f = 0;
layout(constant_id = 8) const float out_batch_head_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint q_idx = gl_GlobalInvocationID.z;

    uint head_size = uint(head_size_f);
    uint kv_seq_len = uint(kv_seq_len_f);
    float scale_val = scale_val_f;
    uint q_seq_head = uint(q_seq_head_f);
    uint q_batch_head = uint(q_batch_head_f);
    uint k_seq_head = uint(k_seq_head_f);
    uint k_batch_head = uint(k_batch_head_f);
    uint out_seq_kv = uint(out_seq_kv_f);
    uint out_batch_head = uint(out_batch_head_f);

    uint q_base = b * q_batch_head + h * q_seq_head + q_idx * head_size;
    uint k_base = b * k_batch_head + h * k_seq_head;
    uint out_idx = b * out_batch_head + h * out_seq_kv + q_idx * kv_seq_len;

    // 计算每个kv位置的attention score
    for (uint kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx) {
        float sum = 0.0;
        uint q_idx_inner = q_base;
        uint k_idx = k_base;
        for (uint d = 0; d < head_size; ++d) {
            sum += q_data[q_idx_inner] * k_data[k_idx];
            q_idx_inner++;
            k_idx++;
        }
        out_data[out_idx] = sum * scale_val;
        k_base += head_size;
        out_idx++;
    }
}
""")
        # 添加bias和应用mask的shader
        self.shader_add_bias_mask = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InBuf { float in_data[]; };
layout(set=0, binding=1) readonly buffer BiasBuf { float bias_data[]; };
layout(set=0, binding=2) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float num_heads_f = 0;
layout(constant_id = 1) const float kv_seq_len_f = 0;
layout(constant_id = 2) const float is_causal_f = 0;
layout(constant_id = 3) const float seq_kv_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint q_idx = gl_GlobalInvocationID.z;

    uint num_heads = uint(num_heads_f);
    uint kv_seq_len = uint(kv_seq_len_f);
    uint is_causal = uint(is_causal_f);
    uint seq_kv = uint(seq_kv_f);

    uint base = b * num_heads * seq_kv + h * seq_kv + q_idx * kv_seq_len;
    uint bias_base = q_idx * kv_seq_len;

    uint idx = base;
    uint bias_idx = bias_base;
    for (uint kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx) {
        float val = in_data[idx] + bias_data[bias_idx];

        // 应用causal mask
        if (is_causal == 1 && kv_idx > q_idx) {
            val = -1.0 / 0.0;  // -inf
        }

        out_data[idx] = val;
        idx++;
        bias_idx++;
    }
}
""")
        # Softcap shader
        self.shader_softcap = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InBuf { float in_data[]; };
layout(set=0, binding=1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float num_heads_f = 0;
layout(constant_id = 1) const float kv_seq_len_f = 0;
layout(constant_id = 2) const float softcap_f = 0;
layout(constant_id = 3) const float seq_kv_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint q_idx = gl_GlobalInvocationID.z;

    uint num_heads = uint(num_heads_f);
    uint kv_seq_len = uint(kv_seq_len_f);
    float softcap = softcap_f;
    uint seq_kv = uint(seq_kv_f);

    uint base = b * num_heads * seq_kv + h * seq_kv + q_idx * kv_seq_len;

    if (softcap > 0.0) {
        uint idx = base;
        for (uint kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx) {
            out_data[idx] = tanh(in_data[idx] / softcap) * softcap;
            idx++;
        }
    }
}
""")
        # Shader for softmax（输出到新缓冲区，避免原地更新）
        self.shader_softmax = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InBuf { float in_data[]; };
layout(set=0, binding=1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float num_heads_f = 0;
layout(constant_id = 1) const float kv_seq_len_f = 0;
layout(constant_id = 2) const float seq_kv_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint q_idx = gl_GlobalInvocationID.z;

    uint num_heads = uint(num_heads_f);
    uint kv_seq_len = uint(kv_seq_len_f);
    uint seq_kv = uint(seq_kv_f);

    uint base = b * num_heads * seq_kv + h * seq_kv + q_idx * kv_seq_len;

    // 查找最大值
    float max_val = in_data[base];
    uint idx = base + 1;
    for (uint i = 1; i < kv_seq_len; ++i) {
        max_val = max(max_val, in_data[idx]);
        idx++;
    }

    // 计算exp和sum
    float sum_exp = 0.0;
    idx = base;
    for (uint i = 0; i < kv_seq_len; ++i) {
        float v = in_data[idx] - max_val;
        sum_exp += exp(v);
        idx++;
    }

    // 归一化并输出
    float inv_sum = 1.0 / sum_exp;
    idx = base;
    for (uint i = 0; i < kv_seq_len; ++i) {
        float v = in_data[idx] - max_val;
        out_data[idx] = exp(v) * inv_sum;
        idx++;
    }
}
""")
        # Shader for softmax @ V
        self.shader_attn_v = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer AttnBuf { float attn_data[]; };
layout(set=0, binding=1) readonly buffer VBuf { float v_data[]; };
layout(set=0, binding=2) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float kv_seq_len_f = 0;
layout(constant_id = 1) const float v_head_size_f = 0;
layout(constant_id = 2) const float attn_batch_head_f = 0;
layout(constant_id = 3) const float attn_seq_kv_f = 0;
layout(constant_id = 4) const float v_batch_head_f = 0;
layout(constant_id = 5) const float v_seq_head_f = 0;
layout(constant_id = 6) const float out_batch_head_f = 0;
layout(constant_id = 7) const float out_seq_head_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint q_idx = gl_GlobalInvocationID.z;

    uint kv_seq_len = uint(kv_seq_len_f);
    uint v_head_size = uint(v_head_size_f);
    uint attn_batch_head = uint(attn_batch_head_f);
    uint attn_seq_kv = uint(attn_seq_kv_f);
    uint v_batch_head = uint(v_batch_head_f);
    uint v_seq_head = uint(v_seq_head_f);
    uint out_batch_head = uint(out_batch_head_f);
    uint out_seq_head = uint(out_seq_head_f);

    // 计算基址
    uint attn_base = b * attn_batch_head + h * attn_seq_kv + q_idx * kv_seq_len;
    uint v_base = b * v_batch_head + h * v_seq_head;
    uint out_idx = b * out_batch_head + h * out_seq_head + q_idx * v_head_size;

    for (uint d = 0; d < v_head_size; ++d) {
        float sum = 0.0;
        uint v_idx = v_base + d;
        uint attn_idx = attn_base;
        for (uint kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx) {
            sum += attn_data[attn_idx] * v_data[v_idx];
            attn_idx++;
            v_idx += v_head_size;
        }
        out_data[out_idx] = sum;
        out_idx++;
    }
}
""")
        # Shader for 3D转换：(batch, num_heads, q_seq_len, v_head_size) -> (batch, q_seq_len, num_heads * v_head_size)
        self.shader_transpose_reshape_3d = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InBuf { float in_data[]; };
layout(set=0, binding=1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float v_head_size_f = 0;
layout(constant_id = 1) const float in_seq_head_f = 0;
layout(constant_id = 2) const float in_batch_head_f = 0;
layout(constant_id = 3) const float out_head_all_f = 0;
layout(constant_id = 4) const float out_batch_seq_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint q_idx = gl_GlobalInvocationID.y;
    uint h = gl_GlobalInvocationID.z;

    uint v_head_size = uint(v_head_size_f);
    uint in_seq_head = uint(in_seq_head_f);
    uint in_batch_head = uint(in_batch_head_f);
    uint out_head_all = uint(out_head_all_f);
    uint out_batch_seq = uint(out_batch_seq_f);

    // 输入layout: (batch, num_heads, q_seq_len, v_head_size)
    uint in_idx = b * in_batch_head + h * in_seq_head + q_idx * v_head_size;
    // 输出layout: (batch, q_seq_len, num_heads * v_head_size)
    uint out_idx = b * out_batch_seq + q_idx * out_head_all + h * v_head_size;

    for (uint d = 0; d < v_head_size; ++d) {
        out_data[out_idx] = in_data[in_idx];
        in_idx++;
        out_idx++;
    }
}
""")
        # 序列维度拼接shader (用于past_key/past_value与K/V的拼接)
        self.shader_concat_seq = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer Past { float past_data[]; };
layout(set=0, binding=1) readonly buffer Current { float current_data[]; };
layout(set=0, binding=2) writeonly buffer Output { float out_data[]; };

layout(constant_id = 0) const float num_heads_f = 0;
layout(constant_id = 1) const float past_seq_len_f = 0;
layout(constant_id = 2) const float current_seq_len_f = 0;
layout(constant_id = 3) const float head_size_f = 0;
layout(constant_id = 4) const float total_seq_len_f = 0;
layout(constant_id = 5) const float seq_head_f = 0;
layout(constant_id = 6) const float past_seq_head_f = 0;
layout(constant_id = 7) const float current_seq_head_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint s = gl_GlobalInvocationID.z;

    uint num_heads = uint(num_heads_f);
    uint past_seq_len = uint(past_seq_len_f);
    uint current_seq_len = uint(current_seq_len_f);
    uint head_size = uint(head_size_f);
    uint seq_head = uint(seq_head_f);
    uint past_seq_head = uint(past_seq_head_f);
    uint current_seq_head = uint(current_seq_head_f);

    uint out_idx = b * num_heads * seq_head + h * seq_head + s * head_size;
    // 如果s < past_seq_len，从past复制；否则从current复制
    if (s < past_seq_len) {
        uint past_idx = b * num_heads * past_seq_head + h * past_seq_head + s * head_size;
        for (uint d = 0; d < head_size; ++d) {
            out_data[out_idx] = past_data[past_idx];
            past_idx++;
            out_idx++;
        }
    } else {
        uint current_s = s - past_seq_len;
        uint current_idx = b * num_heads * current_seq_head + h * current_seq_head + current_s * head_size;
        for (uint d = 0; d < head_size; ++d) {
            out_data[out_idx] = current_data[current_idx];
            current_idx++;
            out_idx++;
        }
    }
}
""")
        # GQA扩展shader
        self.shader_gqa_expand = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) readonly buffer InBuf { float in_data[]; };
layout(set=0, binding=1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float kv_num_heads_f = 0;
layout(constant_id = 1) const float head_size_f = 0;
layout(constant_id = 2) const float reps_f = 0;
layout(constant_id = 3) const float seq_head_size_f = 0;
layout(constant_id = 4) const float q_num_heads_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint kv_h = gl_GlobalInvocationID.y;
    uint s = gl_GlobalInvocationID.z;

    uint kv_num_heads = uint(kv_num_heads_f);
    uint head_size = uint(head_size_f);
    uint reps = uint(reps_f);
    uint seq_head_size = uint(seq_head_size_f);
    uint q_num_heads = uint(q_num_heads_f);

    // 输入位置: (batch, kv_num_heads, seq_len, head_size)
    uint in_base = b * kv_num_heads * seq_head_size + kv_h * seq_head_size + s * head_size;
    uint out_batch_base = b * q_num_heads * seq_head_size;

    // 将这个kv_head复制到多个q_head位置（交错方式）
    uint out_base = out_batch_base + kv_h * seq_head_size + s * head_size;
    uint out_stride = kv_num_heads * seq_head_size;
    for (uint r = 0; r < reps; ++r) {
        uint in_idx = in_base;
        uint out_idx = out_base;
        for (uint d = 0; d < head_size; ++d) {
            out_data[out_idx] = in_data[in_idx];
            in_idx++;
            out_idx++;
        }
        out_base += out_stride;
    }
}
""")
    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"AttentionOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        # 保存mask的dtype用于fuse函数判断
        if len(inputs) > 3 and inputs[3] is not None:
            self._mask_dtype = inputs[3].dtype if isinstance(inputs[3], np.ndarray) else None
        else:
            self._mask_dtype = None

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
        assert 0 <= self.qk_matmul_output_mode <= 3, "qk_matmul_output_mode must between 0-3"
        assert self.is_causal in (0, 1), "is_causal must be 0 or 1"
        # 解析输入
        Q_tensor, Q_shape = input_tensors[0]
        K_tensor, K_shape = input_tensors[1]
        V_tensor, V_shape = input_tensors[2]

        attn_mask = input_tensors[3] if len(input_tensors) > 3 and input_tensors[3] is not None else None
        past_key = input_tensors[4] if len(input_tensors) > 4 and input_tensors[4] is not None else None
        past_value = input_tensors[5] if len(input_tensors) > 5 and input_tensors[5] is not None else None

        input_shape_len = len(Q_shape)
        batch = Q_shape[0]

        # 处理3D输入转换为4D
        if input_shape_len == 3:
            assert self.q_num_heads is not None and self.kv_num_heads is not None
            q_seq_len_3d = Q_shape[1]
            k_seq_len_3d = K_shape[1]
            v_seq_len_3d = V_shape[1]

            Q_shape = [batch, self.q_num_heads, q_seq_len_3d, Q_shape[2] // self.q_num_heads]
            K_shape = [batch, self.kv_num_heads, k_seq_len_3d, K_shape[2] // self.kv_num_heads]
            V_shape = [batch, self.kv_num_heads, v_seq_len_3d, V_shape[2] // self.kv_num_heads]

        assert len(Q_shape) == 4 and len(K_shape) == 4 and len(V_shape) == 4

        q_num_heads = Q_shape[1]
        q_seq_len = Q_shape[2]
        head_size = Q_shape[3]

        kv_num_heads = K_shape[1]
        kv_seq_len = K_shape[2]
        v_head_size = V_shape[3]

        # 计算scale - ONNX规范：
        if self.scale is None:
            scale = 1.0 / (head_size ** 0.5)
        else:
            scale = self.scale

        # 处理past_key和past_value（KV cache）
        if past_key is not None:
            # Concatenate past_key with K along sequence dimension (axis=2)
            past_key_tensor, past_key_shape = past_key
            assert len(past_key_shape) == 4
            past_kv_seq_len = past_key_shape[2]
            total_kv_seq_len = past_kv_seq_len + kv_seq_len

            # 创建拼接后的present_key
            present_key = self.manager.tensor(
                np.zeros(batch * kv_num_heads * total_kv_seq_len * head_size, dtype=np.float32))
            updated_tensors.append(present_key)

            # 预计算stride常量
            seq_head = total_kv_seq_len * head_size
            past_seq_head = past_kv_seq_len * head_size
            current_seq_head = kv_seq_len * head_size

            updated_algorithms.append(self.manager.algorithm(
                [past_key_tensor, K_tensor, present_key],
                self.shader_concat_seq,
                (batch, kv_num_heads, total_kv_seq_len),
                [kv_num_heads, past_kv_seq_len, kv_seq_len, head_size, total_kv_seq_len, seq_head, past_seq_head, current_seq_head],
                []
            ))

            K_tensor = present_key
            K_shape = [batch, kv_num_heads, total_kv_seq_len, head_size]
            present_key_shape = K_shape
        else:
            present_key = K_tensor
            present_key_shape = K_shape

        if past_value is not None:
            # 将 past_value 与 V 沿序列维度（axis=2）进行拼接
            past_value_tensor, past_value_shape = past_value
            assert len(past_value_shape) == 4
            past_kv_seq_len_v = past_value_shape[2]
            total_kv_seq_len_v = past_kv_seq_len_v + V_shape[2]

            # 创建拼接后的present_value
            present_value = self.manager.tensor(
                np.zeros(batch * kv_num_heads * total_kv_seq_len_v * v_head_size, dtype=np.float32))
            updated_tensors.append(present_value)

            # 预计算stride常量
            seq_head_v = total_kv_seq_len_v * v_head_size
            past_seq_head_v = past_kv_seq_len_v * v_head_size
            current_seq_head_v = V_shape[2] * v_head_size

            updated_algorithms.append(self.manager.algorithm(
                [past_value_tensor, V_tensor, present_value],
                self.shader_concat_seq,
                (batch, kv_num_heads, total_kv_seq_len_v),
                [kv_num_heads, past_kv_seq_len_v, V_shape[2], v_head_size, total_kv_seq_len_v, seq_head_v, past_seq_head_v, current_seq_head_v],
                []
            ))

            V_tensor = present_value
            V_shape = [batch, kv_num_heads, total_kv_seq_len_v, v_head_size]
            present_value_shape = V_shape
        else:
            present_value = V_tensor
            present_value_shape = V_shape

        # 更新kv_seq_len
        kv_seq_len = K_shape[2]

        # 处理Group Query Attention
        if q_num_heads != kv_num_heads:
            assert q_num_heads % kv_num_heads == 0, "q_num_heads必须能被kv_num_heads整除"
            seq_reps = q_num_heads // kv_num_heads

            # 扩展K: (batch, kv_num_heads, kv_seq_len, head_size) -> (batch, q_num_heads, kv_seq_len, head_size)
            k_expanded_size = batch * q_num_heads * kv_seq_len * head_size
            K_expanded = self.manager.tensor(np.zeros(k_expanded_size, dtype=np.float32))
            updated_tensors.append(K_expanded)

            # 预计算stride常量
            seq_head_size_k = kv_seq_len * head_size

            updated_algorithms.append(self.manager.algorithm(
                [K_tensor, K_expanded],
                self.shader_gqa_expand,
                (batch, kv_num_heads, kv_seq_len),
                [kv_num_heads, head_size, seq_reps, seq_head_size_k, q_num_heads],
                []
            ))
            # 用于后续计算的K是扩展后的
            K_tensor_for_compute = K_expanded

            # 扩展V: (batch, kv_num_heads, kv_seq_len, v_head_size) -> (batch, q_num_heads, kv_seq_len, v_head_size)
            v_expanded_size = batch * q_num_heads * kv_seq_len * v_head_size
            V_expanded = self.manager.tensor(np.zeros(v_expanded_size, dtype=np.float32))
            updated_tensors.append(V_expanded)

            # 预计算stride常量
            seq_head_size_v = kv_seq_len * v_head_size

            updated_algorithms.append(self.manager.algorithm(
                [V_tensor, V_expanded],
                self.shader_gqa_expand,
                (batch, kv_num_heads, kv_seq_len),
                [kv_num_heads, v_head_size, seq_reps, seq_head_size_v, q_num_heads],
                []
            ))
            # 用于后续计算的V是扩展后的
            V_tensor_for_compute = V_expanded

            # present_key和present_value保持原始的kv_num_heads（不扩展）
            # 它们已经在前面的KV cache处理中设置好了
        else:
            # MHA情况，K和V不需要扩展
            K_tensor_for_compute = K_tensor
            V_tensor_for_compute = V_tensor

        # Step 1: Q*K^T 计算（mode=0时需要保存）
        qk_size = batch * q_num_heads * q_seq_len * kv_seq_len
        qk_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
        updated_tensors.append(qk_tensor)

        # 预计算所有stride常量
        q_seq_head = q_seq_len * head_size
        q_batch_head = q_num_heads * q_seq_head
        k_seq_head = kv_seq_len * head_size
        k_batch_head = q_num_heads * k_seq_head
        out_seq_kv = q_seq_len * kv_seq_len
        out_batch_head = q_num_heads * out_seq_kv

        updated_algorithms.append(self.manager.algorithm(
            [Q_tensor, K_tensor_for_compute, qk_tensor],
            self.shader_qk,
            (batch, q_num_heads, q_seq_len),
            [head_size, kv_seq_len, scale, q_seq_head, q_batch_head, k_seq_head, k_batch_head, out_seq_kv, out_batch_head],
            []
        ))

        # Step 2: 添加bias和mask（mode=1时需要保存）
        qk_bias_mask_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
        updated_tensors.append(qk_bias_mask_tensor)

        attn_bias_size = out_seq_kv
        attn_bias_np = np.zeros(attn_bias_size, dtype=np.float32)

        if attn_mask is not None:
            assert self.is_causal != 1, "is_causal与attn_mask不能同时使用"
            attn_mask_tensor, attn_mask_shape = attn_mask
            attn_mask_data = attn_mask_tensor.data()

            # 处理boolean mask - 使用原始dtype判断
            if self._mask_dtype == np.dtype('bool'):
                # boolean mask: True表示不mask，False表示mask
                # ONNX中boolean mask逻辑取反后，False(0.0)位置设为-inf
                for i in range(attn_bias_size):
                    if attn_mask_data[i] == 0.0:
                        attn_bias_np[i] = float('-inf')
            else:
                # float mask直接赋值
                for i in range(out_seq_kv):
                    attn_bias_np[i] = attn_mask_data[i]

        bias_tensor = self.manager.tensor(attn_bias_np)
        updated_tensors.append(bias_tensor)

        updated_algorithms.append(self.manager.algorithm(
            [qk_tensor, bias_tensor, qk_bias_mask_tensor],
            self.shader_add_bias_mask,
            (batch, q_num_heads, q_seq_len),
            [q_num_heads, kv_seq_len, self.is_causal, out_seq_kv],
            []
        ))

        # Step 3: 应用softcap（如果提供且大于0）（mode=2时需要保存）
        if self.softcap > 0:
            qk_softcap_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
            updated_tensors.append(qk_softcap_tensor)

            updated_algorithms.append(self.manager.algorithm(
                [qk_bias_mask_tensor, qk_softcap_tensor],
                self.shader_softcap,
                (batch, q_num_heads, q_seq_len),
                [q_num_heads, kv_seq_len, self.softcap, out_seq_kv],
                []
            ))
            # softcap的输出作为softmax的输入
            qk_before_softmax = qk_softcap_tensor
        else:
            # 没有softcap，bias_mask的输出直接作为softmax的输入
            qk_before_softmax = qk_bias_mask_tensor

        # Step 4: Softmax（mode=3时需要保存）
        softmax_output_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
        updated_tensors.append(softmax_output_tensor)

        updated_algorithms.append(self.manager.algorithm(
            [qk_before_softmax, softmax_output_tensor],
            self.shader_softmax,
            (batch, q_num_heads, q_seq_len),
            [q_num_heads, kv_seq_len, out_seq_kv],
            []
        ))

        # Step 5: Softmax @ V
        output_4d_tensor = self.manager.tensor(
            np.zeros(batch * q_num_heads * q_seq_len * v_head_size, dtype=np.float32))
        updated_tensors.append(output_4d_tensor)

        # 预计算stride常量
        v_seq_head = kv_seq_len * v_head_size
        v_batch_head = q_num_heads * v_seq_head
        out_seq_head_v = q_seq_len * v_head_size
        out_batch_head_v = q_num_heads * out_seq_head_v

        updated_algorithms.append(self.manager.algorithm(
            [softmax_output_tensor, V_tensor_for_compute, output_4d_tensor],
            self.shader_attn_v,
            (batch, q_num_heads, q_seq_len),
            [kv_seq_len, v_head_size, out_batch_head, out_seq_kv, v_batch_head, v_seq_head, out_batch_head_v, out_seq_head_v],
            []
        ))

        # Step 6: 如果输入是3D，需要转置和reshape输出
        if input_shape_len == 3:
            # (batch, num_heads, q_seq_len, v_head_size) -> (batch, q_seq_len, num_heads * v_head_size)
            output_tensor = self.manager.tensor(
                np.zeros(batch * q_seq_len * q_num_heads * v_head_size, dtype=np.float32))
            updated_tensors.append(output_tensor)

            # 预计算stride常量
            in_seq_head = q_seq_len * v_head_size
            in_batch_head = q_num_heads * in_seq_head
            out_head_all = q_num_heads * v_head_size
            out_batch_seq = q_seq_len * out_head_all

            updated_algorithms.append(self.manager.algorithm(
                [output_4d_tensor, output_tensor],
                self.shader_transpose_reshape_3d,
                (batch, q_seq_len, q_num_heads),
                [v_head_size, in_seq_head, in_batch_head, out_head_all, out_batch_seq],
                []
            ))

            output_shape = [batch, q_seq_len, out_head_all]
        else:
            output_tensor = output_4d_tensor
            output_shape = [batch, q_num_heads, q_seq_len, v_head_size]

        # 根据qk_matmul_output_mode决定qk_output_for_return返回的中间结果
        if self.qk_matmul_output_mode == 0:
            # 返回qk矩阵乘法的输出（未加bias和mask）
            qk_output_for_return = qk_tensor
        elif self.qk_matmul_output_mode == 1:
            # 返回添加bias和mask后,softcap之前的输出
            qk_output_for_return = qk_bias_mask_tensor
        elif self.qk_matmul_output_mode == 2:
            # 返回softcap后（若有），softmax前的输出
            qk_output_for_return = qk_before_softmax
        else: # self.qk_matmul_output_mode == 3
            # 返回softmax后的输出
            qk_output_for_return = softmax_output_tensor

        # 返回输出: output, present_key, present_value, qk_matmul_output
        result = [
            (output_tensor, output_shape),
            (present_key, present_key_shape),
            (present_value, present_value_shape),
            (qk_output_for_return, [batch, q_num_heads, q_seq_len, kv_seq_len])
        ]

        return result

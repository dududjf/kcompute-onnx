import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


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
        self.shader_qk = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly buffer QBuf {{ float q_data[]; }};
layout(std430, set=0, binding=1) readonly buffer KBuf {{ float k_data[]; }};
layout(std430, set=0, binding=2) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set=0, binding=3) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint bh = gl_GlobalInvocationID.x;
    uint q_idx = gl_GlobalInvocationID.y;
    uint kv_idx = gl_GlobalInvocationID.z;

    uint head_size    = params[0];
    uint kv_seq_len   = params[1];
    float scale_val   = uintBitsToFloat(params[2]);
    uint q_seq_head   = params[3];
    uint q_batch_head = params[4];
    uint k_seq_head   = params[5];
    uint k_batch_head = params[6];
    uint out_seq_kv   = params[7];
    uint out_batch_head = params[8];

    uint batch_heads_total = params[9];
    uint q_seq_total       = params[10];

    if (bh >= batch_heads_total || q_idx >= q_seq_total || kv_idx >= kv_seq_len) return;

    uint q_base  = bh * q_seq_head + q_idx * head_size;
    uint k_base  = bh * k_seq_head + kv_idx * head_size;
    uint out_idx = bh * out_seq_kv  + q_idx * kv_seq_len + kv_idx;

    float sum = 0.0;
    for (uint d = 0; d < head_size; ++d) {{
        sum += q_data[q_base + d] * k_data[k_base + d];
    }}
    out_data[out_idx] = sum * scale_val;
}}
""")

        # 添加bias和应用mask的shader - 没有mask的情况
        self.shader_add_bias_mask_no_mask = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set=0, binding=2) readonly  buffer Params {{ uint params[]; }};

void main() {{
    uint bh     = gl_GlobalInvocationID.x;
    uint q_idx  = gl_GlobalInvocationID.y;
    uint kv_idx = gl_GlobalInvocationID.z;

    uint kv_seq_len        = params[0];
    uint is_causal         = params[1];
    uint seq_kv            = params[2];
    uint batch_heads_total = params[3];
    uint q_seq_total       = params[4];

    if (bh >= batch_heads_total || q_idx >= q_seq_total || kv_idx >= kv_seq_len) return;

    uint idx  = bh * seq_kv + q_idx * kv_seq_len + kv_idx;
    float val = in_data[idx];

    if (is_causal == 1u && kv_idx > q_idx) {{
        val = -1.0 / 0.0;  // -inf
    }}

    out_data[idx] = val;
}}
""")

        # 添加bias和应用mask的shader - 有mask的情况
        self.shader_add_bias_mask_with_mask = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer InBuf   {{ float in_data[];   }};
layout(std430, set=0, binding=1) readonly  buffer MaskBuf {{ float mask_data[]; }};
layout(std430, set=0, binding=2) writeonly buffer OutBuf  {{ float out_data[];  }};
layout(std430, set=0, binding=3) readonly  buffer Params  {{ uint params[]; }};

void main() {{
    uint bh     = gl_GlobalInvocationID.x;
    uint q_idx  = gl_GlobalInvocationID.y;
    uint kv_idx = gl_GlobalInvocationID.z;

    uint kv_seq_len        = params[0];
    uint seq_kv            = params[1];
    uint batch_heads_total = params[2];
    uint q_seq_total       = params[3];

    if (bh >= batch_heads_total || q_idx >= q_seq_total || kv_idx >= kv_seq_len) return;

    uint idx      = bh * seq_kv + q_idx * kv_seq_len + kv_idx;
    uint mask_idx = q_idx * kv_seq_len + kv_idx;

    float val      = in_data[idx];
    float mask_val = mask_data[mask_idx];

    if (mask_val == 0.0) {{
        val = -1.0 / 0.0;  // -inf
    }} else {{
        val = val + mask_val;
    }}

    out_data[idx] = val;
}}
""")

        # Softcap shader
        self.shader_softcap = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set=0, binding=2) readonly  buffer Params {{ uint params[]; }};

void main() {{
    uint bh     = gl_GlobalInvocationID.x;
    uint q_idx  = gl_GlobalInvocationID.y;
    uint kv_idx = gl_GlobalInvocationID.z;

    uint  kv_seq_len        = params[0];
    float softcap           = uintBitsToFloat(params[1]);
    uint  seq_kv            = params[2];
    uint  batch_heads_total = params[3];
    uint  q_seq_total       = params[4];

    if (bh >= batch_heads_total || q_idx >= q_seq_total || kv_idx >= kv_seq_len) return;

    uint idx    = bh * seq_kv + q_idx * kv_seq_len + kv_idx;
    out_data[idx] = tanh(in_data[idx] / softcap) * softcap;
}}
""")

        # Softmax shader
        self.shader_softmax = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set=0, binding=2) readonly  buffer Params {{ uint params[]; }};

void main() {{
    uint b     = gl_GlobalInvocationID.x;
    uint h     = gl_GlobalInvocationID.y;
    uint q_idx = gl_GlobalInvocationID.z;

    uint num_heads  = params[0];
    uint kv_seq_len = params[1];
    uint seq_kv     = params[2];
    uint batch      = params[3];
    uint q_seq_len  = params[4];

    if (b >= batch || h >= num_heads || q_idx >= q_seq_len) return;

    uint base = b * num_heads * seq_kv + h * seq_kv + q_idx * kv_seq_len;

    // 查找最大值（数值稳定性）
    float max_val = in_data[base];
    for (uint i = 1; i < kv_seq_len; ++i) {{
        max_val = max(max_val, in_data[base + i]);
    }}

    // 计算 exp 和 sum
    float sum_exp = 0.0;
    for (uint i = 0; i < kv_seq_len; ++i) {{
        sum_exp += exp(in_data[base + i] - max_val);
    }}

    // 归一化并输出
    float inv_sum = 1.0 / sum_exp;
    for (uint i = 0; i < kv_seq_len; ++i) {{
        out_data[base + i] = exp(in_data[base + i] - max_val) * inv_sum;
    }}
}}
""")

        # Softmax @ V shader
        self.shader_attn_v = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer AttnBuf {{ float attn_data[]; }};
layout(std430, set=0, binding=1) readonly  buffer VBuf    {{ float v_data[];    }};
layout(std430, set=0, binding=2) writeonly buffer OutBuf  {{ float out_data[];  }};
layout(std430, set=0, binding=3) readonly  buffer Params  {{ uint params[]; }};

void main() {{
    uint bh     = gl_GlobalInvocationID.x;
    uint q_idx  = gl_GlobalInvocationID.y;
    uint d      = gl_GlobalInvocationID.z;

    uint kv_seq_len        = params[0];
    uint v_head_size       = params[1];
    uint attn_seq_kv       = params[2];
    uint v_seq_head        = params[3];
    uint out_seq_head      = params[4];
    uint batch_heads_total = params[5];
    uint q_seq_len         = params[6];

    if (bh >= batch_heads_total || q_idx >= q_seq_len || d >= v_head_size) return;

    uint attn_base = bh * attn_seq_kv  + q_idx * kv_seq_len;
    uint v_base    = bh * v_seq_head   + d;
    uint out_idx   = bh * out_seq_head + q_idx * v_head_size + d;

    float sum = 0.0;
    for (uint kv = 0; kv < kv_seq_len; ++kv) {{
        sum += attn_data[attn_base + kv] * v_data[v_base + kv * v_head_size];
    }}
    out_data[out_idx] = sum;
}}
""")

        # 3D转换 shader: (batch, num_heads, q_seq_len, v_head_size) -> (batch, q_seq_len, num_heads * v_head_size)
        self.shader_transpose_reshape_3d = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set=0, binding=2) readonly  buffer Params {{ uint params[]; }};

void main() {{
    uint b     = gl_GlobalInvocationID.x;
    uint q_idx = gl_GlobalInvocationID.y;
    uint h     = gl_GlobalInvocationID.z;

    uint v_head_size  = params[0];
    uint in_seq_head  = params[1];
    uint in_batch_head= params[2];
    uint out_head_all = params[3];
    uint out_batch_seq= params[4];
    uint batch        = params[5];
    uint q_seq_len    = params[6];
    uint num_heads    = params[7];

    if (b >= batch || q_idx >= q_seq_len || h >= num_heads) return;

    uint in_idx  = b * in_batch_head + h * in_seq_head  + q_idx * v_head_size;
    uint out_idx = b * out_batch_seq  + q_idx * out_head_all + h * v_head_size;

    for (uint d = 0; d < v_head_size; ++d) {{
        out_data[out_idx + d] = in_data[in_idx + d];
    }}
}}
""")

        # 序列维度拼接 shader (past_key/past_value 与 K/V 的拼接)
        self.shader_concat_seq = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer Past    {{ float past_data[];    }};
layout(std430, set=0, binding=1) readonly  buffer Current {{ float current_data[]; }};
layout(std430, set=0, binding=2) writeonly buffer Output  {{ float out_data[];     }};
layout(std430, set=0, binding=3) readonly  buffer Params  {{ uint params[]; }};

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint s = gl_GlobalInvocationID.z;

    uint num_heads       = params[0];
    uint past_seq_len    = params[1];
    uint current_seq_len = params[2];
    uint head_size       = params[3];
    uint total_seq_len   = params[4];
    uint seq_head        = params[5];
    uint past_seq_head   = params[6];
    uint current_seq_head= params[7];
    uint batch           = params[8];

    if (b >= batch || h >= num_heads || s >= total_seq_len) return;

    uint out_idx = b * num_heads * seq_head + h * seq_head + s * head_size;

    if (s < past_seq_len) {{
        uint past_idx = b * num_heads * past_seq_head + h * past_seq_head + s * head_size;
        for (uint d = 0; d < head_size; ++d) {{
            out_data[out_idx + d] = past_data[past_idx + d];
        }}
    }} else {{
        uint current_s   = s - past_seq_len;
        uint current_idx = b * num_heads * current_seq_head + h * current_seq_head + current_s * head_size;
        for (uint d = 0; d < head_size; ++d) {{
            out_data[out_idx + d] = current_data[current_idx + d];
        }}
    }}
}}
""")

        # GQA 扩展 shader: (batch, kv_num_heads, seq, head) -> (batch, q_num_heads, seq, head)
        # 使用与 np.tile(K, [1, reps, 1, 1]) 一致的 interleaved 布局：
        #   kv_h=0 -> q_h=0, kv_num_heads, 2*kv_num_heads, ...
        #   kv_h=1 -> q_h=1, kv_num_heads+1, 2*kv_num_heads+1, ...
        self.shader_gqa_expand = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set=0, binding=0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set=0, binding=2) readonly  buffer Params {{ uint params[]; }};

void main() {{
    uint b    = gl_GlobalInvocationID.x;
    uint kv_h = gl_GlobalInvocationID.y;
    uint s    = gl_GlobalInvocationID.z;

    uint kv_num_heads  = params[0];
    uint head_size     = params[1];
    uint reps          = params[2];
    uint seq_head_size = params[3];  // seq_len * head_size
    uint q_num_heads   = params[4];
    uint seq_len       = params[5];
    uint batch         = params[6];

    if (b >= batch || kv_h >= kv_num_heads || s >= seq_len) return;

    // 输入: (batch, kv_num_heads, seq_len, head_size)
    uint in_base = b * kv_num_heads * seq_head_size + kv_h * seq_head_size + s * head_size;

    // interleaved: q_h = kv_h, kv_h + kv_num_heads, kv_h + 2*kv_num_heads, ...
    // stride between consecutive replicas along q_head dim = kv_num_heads * seq_head_size
    uint out_base  = b * q_num_heads * seq_head_size + kv_h * seq_head_size + s * head_size;
    uint out_stride = kv_num_heads * seq_head_size;

    for (uint r = 0; r < reps; ++r) {{
        for (uint d = 0; d < head_size; ++d) {{
            out_data[out_base + d] = in_data[in_base + d];
        }}
        out_base += out_stride;
    }}
}}
""")

    def __repr__(self):
        return f"AttentionOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

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
        assert 0 <= self.qk_matmul_output_mode <= 3, "qk_matmul_output_mode must between 0-3"
        assert self.is_causal in (0, 1), "is_causal must be 0 or 1"

        # ---------- 解析输入 ----------
        Q_tensor, Q_shape = input_tensors[0]
        K_tensor, K_shape = input_tensors[1]
        V_tensor, V_shape = input_tensors[2]

        attn_mask  = input_tensors[3] if len(input_tensors) > 3 and input_tensors[3] is not None else None
        past_key   = input_tensors[4] if len(input_tensors) > 4 and input_tensors[4] is not None else None
        past_value = input_tensors[5] if len(input_tensors) > 5 and input_tensors[5] is not None else None

        input_shape_len = len(Q_shape)
        batch = Q_shape[0]

        # ---------- 3D → 4D ----------
        if input_shape_len == 3:
            assert self.q_num_heads is not None and self.kv_num_heads is not None
            Q_shape = [batch, self.q_num_heads, Q_shape[1], Q_shape[2] // self.q_num_heads]
            K_shape = [batch, self.kv_num_heads, K_shape[1], K_shape[2] // self.kv_num_heads]
            V_shape = [batch, self.kv_num_heads, V_shape[1], V_shape[2] // self.kv_num_heads]

        assert len(Q_shape) == 4 and len(K_shape) == 4 and len(V_shape) == 4

        q_num_heads = Q_shape[1]
        q_seq_len   = Q_shape[2]
        head_size   = Q_shape[3]

        kv_num_heads = K_shape[1]
        kv_seq_len   = K_shape[2]
        v_head_size  = V_shape[3]

        scale = self.scale if self.scale is not None else 1.0 / (head_size ** 0.5)

        # ---------- KV Cache：past_key 拼接 ----------
        if past_key is not None:
            past_key_tensor, past_key_shape = past_key
            assert len(past_key_shape) == 4
            past_kv_seq_len   = past_key_shape[2]
            total_kv_seq_len  = past_kv_seq_len + kv_seq_len

            present_key = self.manager.tensor(
                np.zeros(batch * kv_num_heads * total_kv_seq_len * head_size, dtype=np.float32))
            updated_tensors.append(present_key)

            seq_head         = total_kv_seq_len * head_size
            past_seq_head    = past_kv_seq_len  * head_size
            current_seq_head = kv_seq_len       * head_size

            param_concat_k = self.manager.tensor_t(np.array([
                kv_num_heads, past_kv_seq_len, kv_seq_len, head_size,
                total_kv_seq_len, seq_head, past_seq_head, current_seq_head, batch
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_concat_k])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [past_key_tensor, K_tensor, present_key, param_concat_k],
                self.shader_concat_seq,
                (batch, kv_num_heads, total_kv_seq_len)
            ))

            K_tensor          = present_key
            K_shape           = [batch, kv_num_heads, total_kv_seq_len, head_size]
            present_key_shape = K_shape
        else:
            present_key       = K_tensor
            present_key_shape = K_shape

        # ---------- KV Cache：past_value 拼接 ----------
        if past_value is not None:
            past_value_tensor, past_value_shape = past_value
            assert len(past_value_shape) == 4
            past_kv_seq_len_v   = past_value_shape[2]
            total_kv_seq_len_v  = past_kv_seq_len_v + V_shape[2]

            present_value = self.manager.tensor(
                np.zeros(batch * kv_num_heads * total_kv_seq_len_v * v_head_size, dtype=np.float32))
            updated_tensors.append(present_value)

            seq_head_v         = total_kv_seq_len_v * v_head_size
            past_seq_head_v    = past_kv_seq_len_v  * v_head_size
            current_seq_head_v = V_shape[2]         * v_head_size

            param_concat_v = self.manager.tensor_t(np.array([
                kv_num_heads, past_kv_seq_len_v, V_shape[2], v_head_size,
                total_kv_seq_len_v, seq_head_v, past_seq_head_v, current_seq_head_v, batch
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_concat_v])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [past_value_tensor, V_tensor, present_value, param_concat_v],
                self.shader_concat_seq,
                (batch, kv_num_heads, total_kv_seq_len_v)
            ))

            V_tensor            = present_value
            V_shape             = [batch, kv_num_heads, total_kv_seq_len_v, v_head_size]
            present_value_shape = V_shape
        else:
            present_value       = V_tensor
            present_value_shape = V_shape

        # 更新 kv_seq_len（拼接后可能增大）
        kv_seq_len = K_shape[2]

        # ---------- GQA 扩展 ----------
        if q_num_heads != kv_num_heads:
            assert q_num_heads % kv_num_heads == 0, "q_num_heads 必须能被 kv_num_heads 整除"
            seq_reps = q_num_heads // kv_num_heads

            # 扩展 K
            k_expanded_size = batch * q_num_heads * kv_seq_len * head_size
            K_expanded = self.manager.tensor(np.zeros(k_expanded_size, dtype=np.float32))
            updated_tensors.append(K_expanded)

            seq_head_size_k = kv_seq_len * head_size
            param_gqa_k = self.manager.tensor_t(np.array([
                kv_num_heads, head_size, seq_reps, seq_head_size_k, q_num_heads, kv_seq_len, batch
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_gqa_k])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [K_tensor, K_expanded, param_gqa_k],
                self.shader_gqa_expand,
                (batch, kv_num_heads, kv_seq_len)
            ))
            K_tensor_for_compute = K_expanded

            # 扩展 V
            v_expanded_size = batch * q_num_heads * kv_seq_len * v_head_size
            V_expanded = self.manager.tensor(np.zeros(v_expanded_size, dtype=np.float32))
            updated_tensors.append(V_expanded)

            seq_head_size_v = kv_seq_len * v_head_size
            param_gqa_v = self.manager.tensor_t(np.array([
                kv_num_heads, v_head_size, seq_reps, seq_head_size_v, q_num_heads, kv_seq_len, batch
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_gqa_v])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [V_tensor, V_expanded, param_gqa_v],
                self.shader_gqa_expand,
                (batch, kv_num_heads, kv_seq_len)
            ))
            V_tensor_for_compute = V_expanded
        else:
            K_tensor_for_compute = K_tensor
            V_tensor_for_compute = V_tensor

        # ---------- Step 1: Q * K^T ----------
        qk_size   = batch * q_num_heads * q_seq_len * kv_seq_len
        qk_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
        updated_tensors.append(qk_tensor)

        q_seq_head        = q_seq_len   * head_size
        q_batch_head      = q_num_heads * q_seq_head
        k_seq_head        = kv_seq_len  * head_size
        k_batch_head      = q_num_heads * k_seq_head
        out_seq_kv        = q_seq_len   * kv_seq_len
        out_batch_head_qk = q_num_heads * out_seq_kv

        param_qk = self.manager.tensor_t(np.array([
            head_size, kv_seq_len,
            np.float32(scale).view(np.uint32),
            q_seq_head, q_batch_head,
            k_seq_head, k_batch_head,
            out_seq_kv, out_batch_head_qk,
            batch * q_num_heads, q_seq_len
        ], dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_qk])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [Q_tensor, K_tensor_for_compute, qk_tensor, param_qk],
            self.shader_qk,
            (batch * q_num_heads, q_seq_len, kv_seq_len)
        ))

        # ---------- Step 2: 添加 Bias / Mask ----------
        qk_bias_mask_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
        updated_tensors.append(qk_bias_mask_tensor)

        if attn_mask is not None:
            assert self.is_causal != 1, "is_causal 与 attn_mask 不能同时使用"
            attn_mask_tensor, attn_mask_shape = attn_mask

            param_mask = self.manager.tensor_t(np.array([
                kv_seq_len, out_seq_kv,
                batch * q_num_heads, q_seq_len
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_mask])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [qk_tensor, attn_mask_tensor, qk_bias_mask_tensor, param_mask],
                self.shader_add_bias_mask_with_mask,
                (batch * q_num_heads, q_seq_len, kv_seq_len)
            ))
        else:
            param_no_mask = self.manager.tensor_t(np.array([
                kv_seq_len, self.is_causal, out_seq_kv,
                batch * q_num_heads, q_seq_len
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_no_mask])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [qk_tensor, qk_bias_mask_tensor, param_no_mask],
                self.shader_add_bias_mask_no_mask,
                (batch * q_num_heads, q_seq_len, kv_seq_len)
            ))

        # ---------- Step 3: Softcap（可选）----------
        if self.softcap > 0:
            qk_softcap_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
            updated_tensors.append(qk_softcap_tensor)

            param_sc = self.manager.tensor_t(np.array([
                kv_seq_len,
                np.float32(self.softcap).view(np.uint32),
                out_seq_kv,
                batch * q_num_heads, q_seq_len
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_sc])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [qk_bias_mask_tensor, qk_softcap_tensor, param_sc],
                self.shader_softcap,
                (batch * q_num_heads, q_seq_len, kv_seq_len)
            ))
            qk_before_softmax = qk_softcap_tensor
        else:
            qk_before_softmax = qk_bias_mask_tensor

        # ---------- Step 4: Softmax ----------
        softmax_output_tensor = self.manager.tensor(np.zeros(qk_size, dtype=np.float32))
        updated_tensors.append(softmax_output_tensor)

        param_sm = self.manager.tensor_t(np.array([
            q_num_heads, kv_seq_len, out_seq_kv, batch, q_seq_len
        ], dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_sm])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [qk_before_softmax, softmax_output_tensor, param_sm],
            self.shader_softmax,
            (batch, q_num_heads, q_seq_len)
        ))

        # ---------- Step 5: Softmax @ V ----------
        output_4d_tensor = self.manager.tensor(
            np.zeros(batch * q_num_heads * q_seq_len * v_head_size, dtype=np.float32))
        updated_tensors.append(output_4d_tensor)

        v_seq_head       = kv_seq_len * v_head_size
        out_seq_head_v   = q_seq_len  * v_head_size

        param_av = self.manager.tensor_t(np.array([
            kv_seq_len, v_head_size,
            out_seq_kv, v_seq_head, out_seq_head_v,
            batch * q_num_heads, q_seq_len
        ], dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_av])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [softmax_output_tensor, V_tensor_for_compute, output_4d_tensor, param_av],
            self.shader_attn_v,
            (batch * q_num_heads, q_seq_len, v_head_size)
        ))

        # ---------- Step 6: 3D 转置/reshape（仅当输入为 3D 时）----------
        if input_shape_len == 3:
            output_tensor = self.manager.tensor(
                np.zeros(batch * q_seq_len * q_num_heads * v_head_size, dtype=np.float32))
            updated_tensors.append(output_tensor)

            in_seq_head   = q_seq_len   * v_head_size
            in_batch_head = q_num_heads * in_seq_head
            out_head_all  = q_num_heads * v_head_size
            out_batch_seq = q_seq_len   * out_head_all

            param_tr = self.manager.tensor_t(np.array([
                v_head_size, in_seq_head, in_batch_head,
                out_head_all, out_batch_seq,
                batch, q_seq_len, q_num_heads
            ], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_tr])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [output_4d_tensor, output_tensor, param_tr],
                self.shader_transpose_reshape_3d,
                (batch, q_seq_len, q_num_heads)
            ))

            output_shape = [batch, q_seq_len, out_head_all]
        else:
            output_tensor = output_4d_tensor
            output_shape  = [batch, q_num_heads, q_seq_len, v_head_size]

        # ---------- qk_matmul_output_mode 决定返回哪个中间结果 ----------
        if self.qk_matmul_output_mode == 0:
            qk_output_for_return = qk_tensor
        elif self.qk_matmul_output_mode == 1:
            qk_output_for_return = qk_bias_mask_tensor
        elif self.qk_matmul_output_mode == 2:
            qk_output_for_return = qk_before_softmax
        else:  # mode == 3
            qk_output_for_return = softmax_output_tensor

        return [
            (output_tensor,        output_shape),
            (present_key,          present_key_shape),
            (present_value,        present_value_shape),
            (qk_output_for_return, [batch, q_num_heads, q_seq_len, kv_seq_len]),
        ]

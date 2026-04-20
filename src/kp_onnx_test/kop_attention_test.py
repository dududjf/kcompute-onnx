from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_attention import AttentionOp


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s


def _softcap(X, softcap):
    if softcap > 0:
        Y = X / softcap
        Y = np.tanh(Y)
        return Y * softcap
    else:
        return X


def onnx_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    attn_mask: np.ndarray = None,
    past_key: np.ndarray = None,
    past_value: np.ndarray = None,
    scale=None,
    is_causal=0,
    q_num_heads=None,
    kv_num_heads=None,
    softmax_precision=None,
    softcap=None,
    qk_matmul_output_mode=0,
):
    """ONNX Attention参考实现"""
    assert len(Q.shape) == len(K.shape) == len(V.shape)
    input_shape_len = len(Q.shape)
    batch_size = Q.shape[0]

    if len(Q.shape) == 3:
        hidden_size_q = Q.shape[2]
        hidden_size_k = K.shape[2]
        hidden_size_v = V.shape[2]
        assert q_num_heads is not None and kv_num_heads is not None

        head_size_q = hidden_size_q // q_num_heads
        new_shape_q = [batch_size, q_num_heads, Q.shape[1], head_size_q]
        Q = np.reshape(Q, new_shape_q)

        head_size_k = hidden_size_k // kv_num_heads
        new_shape_k = [batch_size, kv_num_heads, K.shape[1], head_size_k]
        K = np.reshape(K, new_shape_k)

        head_size_v = hidden_size_v // kv_num_heads
        new_shape_v = [batch_size, kv_num_heads, V.shape[1], head_size_v]
        V = np.reshape(V, new_shape_v)

    assert len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4

    if scale is None:
        q_head_size = Q.shape[3]
        scale = 1 / np.sqrt(q_head_size)
    scale = np.sqrt(scale)

    if past_key is not None:
        present_key = np.concatenate((past_key, K), axis=2)
    else:
        present_key = K
    if past_value is not None:
        present_value = np.concatenate((past_value, V), axis=2)
    else:
        present_value = V
    K = present_key
    V = present_value

    q_sequence_length = Q.shape[2]
    kv_sequence_length = K.shape[2]
    attn_bias = np.zeros((q_sequence_length, kv_sequence_length), dtype=Q.dtype)

    if is_causal == 1:
        assert attn_mask is None
        temp_mask = np.ones((q_sequence_length, kv_sequence_length), dtype=bool)
        temp_mask = np.tril(temp_mask, k=0)
        temp_mask = np.logical_not(temp_mask)
        attn_bias_ma = np.ma.array(attn_bias, mask=temp_mask)
        attn_bias = attn_bias_ma.filled(fill_value=float("-inf"))

    if attn_mask is not None:
        assert is_causal != 1
        if attn_mask.dtype == bool:
            attn_mask = np.logical_not(attn_mask)
            attn_bias_ma = np.ma.array(attn_bias, mask=attn_mask)
            attn_bias = attn_bias_ma.filled(fill_value=float("-inf"))
        else:
            attn_bias += attn_mask

    if q_num_heads is None:
        q_num_heads = Q.shape[1]
    if kv_num_heads is None:
        k_num_heads = K.shape[1]
        v_num_heads = K.shape[1]
    else:
        k_num_heads = kv_num_heads
        v_num_heads = kv_num_heads

    if (
        (q_num_heads != k_num_heads)
        and (q_num_heads % k_num_heads == 0)
        and (k_num_heads == v_num_heads)
    ):
        seq_reps = q_num_heads // k_num_heads
        reps = [1, seq_reps, 1, 1]
        K = np.tile(K, reps)
        V = np.tile(V, reps)

    k_transpose = np.transpose(K, (0, 1, 3, 2))
    qk_matmul_output = np.matmul(Q * scale, k_transpose * scale)
    qk_with_bias = qk_matmul_output + attn_bias
    if qk_matmul_output_mode == 1:
        qk_matmul_output = qk_matmul_output + attn_bias

    if softcap is not None:
        qk_with_bias = _softcap(qk_with_bias, softcap)
        if qk_matmul_output_mode == 2:
            qk_matmul_output = qk_with_bias

    qk_softmax = _softmax(qk_with_bias)
    if qk_matmul_output_mode == 3:
        qk_matmul_output = qk_softmax
    qk_matmul_output = qk_matmul_output.astype(Q.dtype)

    output = np.matmul(qk_softmax, V).astype(Q.dtype)
    if input_shape_len == 3:
        output = np.transpose(output, (0, 2, 1, 3))
        output = output.reshape(output.shape[0], output.shape[1], -1)

    return output, present_key, present_value, qk_matmul_output


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
print()

# Case 1: 基础4D输入（标准MHA）
print("Case 1: 基础4D输入（标准MHA）")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 2: 3D输入（需要reshape）
print("Case 2: 3D输入（需要reshape）")
batch = 2
q_seq_len = 64
kv_seq_len = 64
q_num_heads = 8
kv_num_heads = 8
head_size = 32
hidden_size = q_num_heads * head_size

Q = np.random.randn(batch, q_seq_len, hidden_size).astype(np.float32)
K = np.random.randn(batch, kv_seq_len, hidden_size).astype(np.float32)
V = np.random.randn(batch, kv_seq_len, hidden_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 3: 带float attention mask
print("Case 3: 带float attention mask")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
attn_mask = np.random.randn(q_seq_len, kv_seq_len).astype(np.float32) * 0.1

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, attn_mask=attn_mask)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, attn_mask)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 4: 带boolean attention mask
print("Case 4: 带boolean attention mask")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
# True表示不mask，False表示mask
attn_mask = np.random.rand(q_seq_len, kv_seq_len) > 0.3
attn_mask = attn_mask.astype(bool)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, attn_mask=attn_mask)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, attn_mask)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 5: is_causal=1（因果mask）
print("Case 5: is_causal=1（因果mask）")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, is_causal=1)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, is_causal=1)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 6: GQA（Group Query Attention）
print("Case 6: GQA（Group Query Attention）")
batch = 2
q_num_heads = 8
kv_num_heads = 2
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, q_num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, kv_num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, kv_num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 7: 带softcap
print("Case 7: 带softcap")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32
softcap = 50.0

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, softcap=softcap)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, softcap=softcap)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 8: 自定义scale
print("Case 8: 自定义scale")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32
scale = 0.125

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, scale=scale)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, scale=scale)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 9: qk_matmul_output_mode=1（softmax前保存qk）
print("Case 9: qk_matmul_output_mode=1（softmax前保存qk）")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, qk_matmul_output_mode=1)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, qk_matmul_output_mode=1)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 10: qk_matmul_output_mode=2（softmax+softcap后保存qk）
print("Case 10: qk_matmul_output_mode=2（softmax+softcap后保存qk）")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32
softcap = 50.0

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, softcap=softcap, qk_matmul_output_mode=2)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, softcap=softcap, qk_matmul_output_mode=2)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 11: qk_matmul_output_mode=3（softmax后保存qk）
print("Case 11: qk_matmul_output_mode=3（softmax后保存qk）")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, qk_matmul_output_mode=3)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, qk_matmul_output_mode=3)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 12: 带 KV cache
print("Case 12: 带past_key和past_value（KV cache）")
batch = 2
num_heads = 8
q_seq_len = 16  # 新的query序列
past_kv_seq_len = 48  # 过去的KV序列
current_kv_seq_len = 16  # 当前的KV序列
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
past_key = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)
past_value = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(Q, K, V, past_key=past_key, past_value=past_value)
print("NumPy:", np_out.shape, "present_key:", np_present_key.shape, "present_value:", np_present_value.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, None, past_key, past_value)
print(f"{attention_op}:", kp_out.shape, "present_key:", kp_present_key.shape, "present_value:", kp_present_value.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("present_key shape equal:", kp_present_key.shape == np_present_key.shape)
print("present_key Max error:", np.abs(np_present_key - kp_present_key).max())
print("present_key All close:", np.allclose(np_present_key, kp_present_key, rtol=1e-5, atol=1e-5))
print("present_value shape equal:", kp_present_value.shape == np_present_value.shape)
print("present_value Max error:", np.abs(np_present_value - kp_present_value).max())
print("present_value All close:", np.allclose(np_present_value, kp_present_value, rtol=1e-5, atol=1e-5))
print("----")

# Case 13: KV cache + GQA
print("Case 13: KV cache + GQA")
batch = 2
q_num_heads = 8
kv_num_heads = 2
q_seq_len = 16
past_kv_seq_len = 48
current_kv_seq_len = 16
head_size = 32

Q = np.random.randn(batch, q_num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, kv_num_heads, current_kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, kv_num_heads, current_kv_seq_len, head_size).astype(np.float32)
past_key = np.random.randn(batch, kv_num_heads, past_kv_seq_len, head_size).astype(np.float32)
past_value = np.random.randn(batch, kv_num_heads, past_kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, past_key=past_key, past_value=past_value,
    q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
)
print("NumPy:", np_out.shape, "present_key:", np_present_key.shape, "present_value:", np_present_value.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, None, past_key, past_value)
print(f"{attention_op}:", kp_out.shape, "present_key:", kp_present_key.shape, "present_value:", kp_present_value.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("present_key shape equal:", kp_present_key.shape == np_present_key.shape)
print("present_key Max error:", np.abs(np_present_key - kp_present_key).max())
print("present_key All close:", np.allclose(np_present_key, kp_present_key, rtol=1e-5, atol=1e-5))
print("present_value shape equal:", kp_present_value.shape == np_present_value.shape)
print("present_value Max error:", np.abs(np_present_value - kp_present_value).max())
print("present_value All close:", np.allclose(np_present_value, kp_present_value, rtol=1e-5, atol=1e-5))
print("----")

# Case 14: KV cache + causal mask
print("Case 14: KV cache + causal mask")
batch = 2
num_heads = 8
q_seq_len = 16
past_kv_seq_len = 48
current_kv_seq_len = 16
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
past_key = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)
past_value = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, past_key=past_key, past_value=past_value, is_causal=1
)
print("NumPy:", np_out.shape, "present_key:", np_present_key.shape, "present_value:", np_present_value.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, is_causal=1)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, None, past_key, past_value)
print(f"{attention_op}:", kp_out.shape, "present_key:", kp_present_key.shape, "present_value:", kp_present_value.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("present_key shape equal:", kp_present_key.shape == np_present_key.shape)
print("present_key Max error:", np.abs(np_present_key - kp_present_key).max())
print("present_key All close:", np.allclose(np_present_key, kp_present_key, rtol=1e-5, atol=1e-5))
print("present_value shape equal:", kp_present_value.shape == np_present_value.shape)
print("present_value Max error:", np.abs(np_present_value - kp_present_value).max())
print("present_value All close:", np.allclose(np_present_value, kp_present_value, rtol=1e-5, atol=1e-5))
print("----")

# Case 15: qk_matmul_output_mode=1 + 3D输入
print("Case 15: qk_matmul_output_mode=1 + 3D输入")
batch = 2
q_seq_len = 64
kv_seq_len = 64
q_num_heads = 8
kv_num_heads = 8
head_size = 32
hidden_size = q_num_heads * head_size

Q = np.random.randn(batch, q_seq_len, hidden_size).astype(np.float32)
K = np.random.randn(batch, kv_seq_len, hidden_size).astype(np.float32)
V = np.random.randn(batch, kv_seq_len, hidden_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads, qk_matmul_output_mode=1
)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads, qk_matmul_output_mode=1)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 16: qk_matmul_output_mode=2 + softcap + GQA
print("Case 16: qk_matmul_output_mode=2 + softcap + GQA")
batch = 2
q_num_heads = 8
kv_num_heads = 2
q_seq_len = 64
kv_seq_len = 64
head_size = 32
softcap = 50.0

Q = np.random.randn(batch, q_num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, kv_num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, kv_num_heads, kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads,
    softcap=softcap, qk_matmul_output_mode=2
)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads,
                           softcap=softcap, qk_matmul_output_mode=2)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 17: qk_matmul_output_mode=1 + KV cache
print("Case 17: qk_matmul_output_mode=1 + KV cache")
batch = 2
num_heads = 8
q_seq_len = 16
past_kv_seq_len = 48
current_kv_seq_len = 16
head_size = 32

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
past_key = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)
past_value = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, past_key=past_key, past_value=past_value, qk_matmul_output_mode=1
)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, qk_matmul_output_mode=1)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, None, past_key, past_value)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 18: qk_matmul_output_mode=2 + KV cache + causal mask
print("Case 18: qk_matmul_output_mode=2 + KV cache + causal mask")
batch = 2
num_heads = 8
q_seq_len = 16
past_kv_seq_len = 48
current_kv_seq_len = 16
head_size = 32
softcap = 50.0

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, current_kv_seq_len, head_size).astype(np.float32)
past_key = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)
past_value = np.random.randn(batch, num_heads, past_kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, past_key=past_key, past_value=past_value, is_causal=1,
    softcap=softcap, qk_matmul_output_mode=2
)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, is_causal=1, softcap=softcap, qk_matmul_output_mode=2)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, None, past_key, past_value)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 19: qk_matmul_output_mode=3 + KV cache + GQA + softcap
print("Case 19: qk_matmul_output_mode=3 + KV cache + GQA + softcap")
batch = 2
q_num_heads = 8
kv_num_heads = 2
q_seq_len = 16
past_kv_seq_len = 48
current_kv_seq_len = 16
head_size = 32
softcap = 50.0

Q = np.random.randn(batch, q_num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, kv_num_heads, current_kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, kv_num_heads, current_kv_seq_len, head_size).astype(np.float32)
past_key = np.random.randn(batch, kv_num_heads, past_kv_seq_len, head_size).astype(np.float32)
past_value = np.random.randn(batch, kv_num_heads, past_kv_seq_len, head_size).astype(np.float32)

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, past_key=past_key, past_value=past_value, q_num_heads=q_num_heads,
    kv_num_heads=kv_num_heads, softcap=softcap, qk_matmul_output_mode=3
)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads,
                           softcap=softcap, qk_matmul_output_mode=3)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, None, past_key, past_value)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")

# Case 20: qk_matmul_output_mode=1 + float mask + softcap
print("Case 20: qk_matmul_output_mode=1 + float mask + softcap")
batch = 2
num_heads = 8
q_seq_len = 64
kv_seq_len = 64
head_size = 32
softcap = 50.0

Q = np.random.randn(batch, num_heads, q_seq_len, head_size).astype(np.float32)
K = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
V = np.random.randn(batch, num_heads, kv_seq_len, head_size).astype(np.float32)
attn_mask = np.random.randn(q_seq_len, kv_seq_len).astype(np.float32) * 0.1

t0 = time.time()
np_out, np_present_key, np_present_value, np_qk = onnx_attention(
    Q, K, V, attn_mask=attn_mask, softcap=softcap, qk_matmul_output_mode=1
)
print("NumPy:", np_out.shape, "qk_output:", np_qk.shape, time.time() - t0, "seconds")

t0 = time.time()
attention_op = AttentionOp(mgr, softcap=softcap, qk_matmul_output_mode=1)
kp_out, kp_present_key, kp_present_value, kp_qk = attention_op.run(Q, K, V, attn_mask)
print(f"{attention_op}:", kp_out.shape, "qk_output:", kp_qk.shape, time.time() - t0, "seconds")

print("output shape equal:", kp_out.shape == np_out.shape)
print("output Max error:", np.abs(np_out - kp_out).max())
print("output All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("qk_output shape equal:", kp_qk.shape == np_qk.shape)
print("qk_output Max error:", np.abs(np_qk - kp_qk).max())
print("qk_output All close:", np.allclose(np_qk, kp_qk, rtol=1e-4, atol=1e-4))
print("----")


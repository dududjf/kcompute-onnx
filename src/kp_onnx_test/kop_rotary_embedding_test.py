from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_rotary_embedding import RotaryEmbeddingOp


def rotary_embedding(
    input: np.ndarray,
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    position_ids: np.ndarray = None,
    interleaved=None,
    rotary_embedding_dim=None,
    num_heads=None,
) -> np.ndarray:
    """ONNX RotaryEmbedding reference implementation"""
    original_input_shape = input.shape
    # First ensure input to be processed has shape [batch_size, seq_len, num_heads, head_size]
    if len(input.shape) == 4:
        input = np.transpose(input, (0, 2, 1, 3))
    batch_size = input.shape[0]
    sequence_length = input.shape[1]
    if len(input.shape) == 3:
        hidden_size = input.shape[2]
        assert num_heads != 0
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        input = np.reshape(input, new_shape)
    assert len(input.shape) == 4
    head_size = input.shape[3]

    # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
    if rotary_embedding_dim is None or rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = input[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)

    # Retrieve sin and cos caches using position ids
    if position_ids is not None:
        cos = cos_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, head_size/2]
        sin = sin_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, head_size/2]
    else:
        cos = cos_cache
        sin = sin_cache
    cos = cos[
        :, :, :rotary_embedding_dim_half
    ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    sin = sin[
        :, :, :rotary_embedding_dim_half
    ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    cos = np.expand_dims(
        cos, axis=2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    sin = np.expand_dims(
        sin, axis=2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

    # Either divide the input in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = np.split(x_rotate, 2, axis=-1)

    # Calculate real and imaginary values
    real = (cos * x1) - (sin * x2)
    imag = (sin * x1) + (cos * x2)

    # Inserted rotated embeddings back to the original input
    if interleaved:
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        x_rotate_concat = np.concatenate((real, imag), axis=-1)
        x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = np.concatenate((real, imag), axis=-1)
    output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    if len(original_input_shape) == 3:
        output = np.reshape(output, original_input_shape)
    else:
        output = np.transpose(output, (0, 2, 1, 3))
    return output


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
print()

# Case 1: 3D input, non-interleaved, no position_ids, full rotation
print("Case 1: 3D input, non-interleaved, no position_ids, full rotation")
batch = 2
seq_len = 32
num_heads = 8
head_size = 64
hidden_size = num_heads * head_size

input_data = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
cos_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, num_heads=num_heads, interleaved=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=0, num_heads=num_heads)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache)[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 2: 4D input, non-interleaved, no position_ids, full rotation
print("Case 2: 4D input, non-interleaved, no position_ids, full rotation")
batch = 2
num_heads = 8
seq_len = 32
head_size = 64

input_data = np.random.randn(batch, num_heads, seq_len, head_size).astype(np.float32)
cos_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, interleaved=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=0)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache)[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 3: 3D input, interleaved, no position_ids, full rotation
print("Case 3: 3D input, interleaved, no position_ids, full rotation")
batch = 2
seq_len = 32
num_heads = 8
head_size = 64
hidden_size = num_heads * head_size

input_data = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
cos_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, num_heads=num_heads, interleaved=1)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=1, num_heads=num_heads)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache)[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 4: 4D input, interleaved, no position_ids, full rotation
print("Case 4: 4D input, interleaved, no position_ids, full rotation")
batch = 2
num_heads = 8
seq_len = 32
head_size = 64

input_data = np.random.randn(batch, num_heads, seq_len, head_size).astype(np.float32)
cos_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, interleaved=1)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=1)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache)[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 5: 3D input, non-interleaved, with position_ids, full rotation
print("Case 5: 3D input, non-interleaved, with position_ids, full rotation")
batch = 2
seq_len = 32
num_heads = 8
head_size = 64
hidden_size = num_heads * head_size
max_seq_len = 128

input_data = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
cos_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
position_ids = np.random.randint(0, max_seq_len, (batch, seq_len)).astype(np.int64)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, position_ids=position_ids, num_heads=num_heads, interleaved=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=0, num_heads=num_heads)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache, position_ids.astype(np.float32))[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 6: 4D input, non-interleaved, with position_ids, full rotation
print("Case 6: 4D input, non-interleaved, with position_ids, full rotation")
batch = 2
num_heads = 8
seq_len = 32
head_size = 64
max_seq_len = 128

input_data = np.random.randn(batch, num_heads, seq_len, head_size).astype(np.float32)
cos_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
position_ids = np.random.randint(0, max_seq_len, (batch, seq_len)).astype(np.int64)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, position_ids=position_ids, interleaved=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=0)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache, position_ids.astype(np.float32))[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 7: 3D input, non-interleaved, no position_ids, partial rotation
print("Case 7: 3D input, non-interleaved, no position_ids, partial rotation")
batch = 2
seq_len = 32
num_heads = 8
head_size = 64
hidden_size = num_heads * head_size
rotary_dim = 32  # Partial rotation

input_data = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
cos_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, num_heads=num_heads, interleaved=0, rotary_embedding_dim=rotary_dim)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=0, num_heads=num_heads, rotary_embedding_dim=rotary_dim)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache)[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 8: 4D input, non-interleaved, no position_ids, partial rotation
print("Case 8: 4D input, non-interleaved, no position_ids, partial rotation")
batch = 2
num_heads = 8
seq_len = 32
head_size = 64
rotary_dim = 32  # Partial rotation

input_data = np.random.randn(batch, num_heads, seq_len, head_size).astype(np.float32)
cos_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(batch, seq_len, head_size // 2).astype(np.float32)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, interleaved=0, rotary_embedding_dim=rotary_dim)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=0, rotary_embedding_dim=rotary_dim)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache)[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 9: 3D input, interleaved, with position_ids, partial rotation
print("Case 9: 3D input, interleaved, with position_ids, partial rotation")
batch = 2
seq_len = 32
num_heads = 8
head_size = 64
hidden_size = num_heads * head_size
max_seq_len = 128
rotary_dim = 48

input_data = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
cos_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
position_ids = np.random.randint(0, max_seq_len, (batch, seq_len)).astype(np.int64)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, position_ids=position_ids, num_heads=num_heads, interleaved=1, rotary_embedding_dim=rotary_dim)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=1, num_heads=num_heads, rotary_embedding_dim=rotary_dim)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache, position_ids.astype(np.float32))[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 10: 4D input, interleaved, with position_ids, partial rotation
print("Case 10: 4D input, interleaved, with position_ids, partial rotation")
batch = 2
num_heads = 8
seq_len = 32
head_size = 64
max_seq_len = 128
rotary_dim = 48

input_data = np.random.randn(batch, num_heads, seq_len, head_size).astype(np.float32)
cos_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
sin_cache = np.random.randn(max_seq_len, head_size // 2).astype(np.float32)
position_ids = np.random.randint(0, max_seq_len, (batch, seq_len)).astype(np.int64)

t0 = time.time()
np_out = rotary_embedding(input_data, cos_cache, sin_cache, position_ids=position_ids, interleaved=1, rotary_embedding_dim=rotary_dim)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
rotary_op = RotaryEmbeddingOp(mgr, interleaved=1, rotary_embedding_dim=rotary_dim)
kp_out = rotary_op.run(input_data, cos_cache, sin_cache, position_ids.astype(np.float32))[0]
print(f"{rotary_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


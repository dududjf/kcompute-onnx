from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_sequence_insert import SequenceInsertOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

seq_insert_op = SequenceInsertOp(mgr)


def onnx_sequence_insert_reference(sequence, tensor, position=None):
    """ONNX reference implementation of SequenceInsert"""
    # Make a copy of input sequence
    seq = []
    if sequence is not None and (
        not isinstance(sequence, np.ndarray) or len(sequence.shape) > 0
    ):
        seq.extend(sequence)

    if position is not None:
        # Handle position as array or scalar
        if isinstance(position, np.ndarray):
            if position.size == 1:
                pos = int(position.flat[0])
            else:
                pos = int(position[0])
        else:
            pos = int(position)
        insert_position = (pos + len(seq)) % max(len(seq), 1)
        seq.insert(insert_position, tensor)
    else:
        # Default position of insertion is at the end of the sequence
        seq.append(tensor)
    return seq


# Case 1: Insert tensor into middle of sequence with explicit position
print("\nCase 1: Insert tensor into middle of sequence at position 1")

# Create initial sequence with 3 tensors
x1 = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
x2 = np.random.uniform(-1, 1, (4, 5)).astype(np.float32)
x3 = np.random.uniform(-1, 1, (3,)).astype(np.float32)
sequence = [x1, x2, x3]

# Tensor to insert
tensor_to_insert = np.random.uniform(-1, 1, (6, 2)).astype(np.float32)

# Position to insert at
position = np.array([1], dtype=np.int64)

print(f"Initial sequence length: {len(sequence)}")
print(f"Initial sequence shapes: {[x.shape for x in sequence]}")
print(f"Tensor to insert shape: {tensor_to_insert.shape}")
print(f"Insert position: {position[0]}")

# NumPy reference
t0 = time.time()
numpy_out = onnx_sequence_insert_reference(sequence, tensor_to_insert, position)
print(f"NumPy: {time.time() - t0} seconds")

# Kompute implementation
t1 = time.time()
kp_out = seq_insert_op.run(sequence, tensor_to_insert, position)
print(f"{seq_insert_op}: {time.time() - t1} seconds")

# Verify results
print(f"\nOutput sequence length: NumPy={len(numpy_out)}, Kompute={len(kp_out)}")
print(f"Expected length: {len(sequence) + 1}")

print("\nOutput sequence shapes:")
for i in range(len(numpy_out)):
    print(f"  Position {i}: NumPy={numpy_out[i].shape}, Kompute={kp_out[i].shape}")

# Check if all tensors match
all_close = True
for i, (numpy_tensor, kp_tensor) in enumerate(zip(numpy_out, kp_out)):
    is_close = np.allclose(numpy_tensor, kp_tensor, rtol=1e-4, atol=1e-4)
    print(f"Tensor {i} all close: {is_close}")
    all_close = all_close and is_close

print(f"\nAll tensors match: {all_close}")

# Verify insertion order
print(f"\nVerify insertion: Tensor at position 1 should be the inserted tensor")
print(f"Match with inserted tensor: {np.allclose(kp_out[1], tensor_to_insert, rtol=1e-4, atol=1e-4)}")

# Case 2: 不传 position
print("\nCase 2: Insert tensor at end of sequence without position")
sequence2 = [x1, x2, x3]
tensor_to_insert2 = np.random.uniform(-1, 1, (5, 4)).astype(np.float32)

t0 = time.time()
numpy_out2 = onnx_sequence_insert_reference(sequence2, tensor_to_insert2)  # position=None
print(f"NumPy: {time.time() - t0} seconds")

t1 = time.time()
kp_out2 = seq_insert_op.run(sequence2, tensor_to_insert2)                  # 不传第3个参数
print(f"{seq_insert_op}: {time.time() - t1} seconds")

print(f"Output length: NumPy={len(numpy_out2)}, Kompute={len(kp_out2)}, Expected={len(sequence2)+1}")
all_close2 = all(np.allclose(n, k, rtol=1e-4, atol=1e-4) for n, k in zip(numpy_out2, kp_out2))
print(f"All tensors match: {all_close2}")
print(f"Last tensor is inserted tensor: {np.allclose(kp_out2[-1], tensor_to_insert2, rtol=1e-4, atol=1e-4)}")

# Case 3: position 为 Python int（非 ndarray）
print("\nCase 3: position is a plain Python int")
sequence3 = [x1, x2, x3]
tensor_to_insert3 = np.random.uniform(-1, 1, (2, 2)).astype(np.float32)
position3 = 2

t0 = time.time()
numpy_out3 = onnx_sequence_insert_reference(sequence3, tensor_to_insert3, position3)
print(f"NumPy: {time.time() - t0} seconds")

t1 = time.time()
kp_out3 = seq_insert_op.run(sequence3, tensor_to_insert3, position3)
print(f"{seq_insert_op}: {time.time() - t1} seconds")

print(f"Output length: NumPy={len(numpy_out3)}, Kompute={len(kp_out3)}, Expected={len(sequence3)+1}")
all_close3 = all(np.allclose(n, k, rtol=1e-4, atol=1e-4) for n, k in zip(numpy_out3, kp_out3))
print(f"All tensors match: {all_close3}")
print(f"Tensor at position 2 is inserted tensor: {np.allclose(kp_out3[2], tensor_to_insert3, rtol=1e-4, atol=1e-4)}")

# Case 4: position 为 size>1 的 ndarray
print("\nCase 4: position is ndarray with size>1")
sequence4 = [x1, x2, x3]
tensor_to_insert4 = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
position4 = np.array([1, 0], dtype=np.int64)   # size=2，取 pos_data[0]=1

t0 = time.time()
numpy_out4 = onnx_sequence_insert_reference(sequence4, tensor_to_insert4, position4)
print(f"NumPy: {time.time() - t0} seconds")

t1 = time.time()
kp_out4 = seq_insert_op.run(sequence4, tensor_to_insert4, position4)
print(f"{seq_insert_op}: {time.time() - t1} seconds")

print(f"Output length: NumPy={len(numpy_out4)}, Kompute={len(kp_out4)}, Expected={len(sequence4)+1}")
all_close4 = all(np.allclose(n, k, rtol=1e-4, atol=1e-4) for n, k in zip(numpy_out4, kp_out4))
print(f"All tensors match: {all_close4}")
print(f"Tensor at position 1 is inserted tensor: {np.allclose(kp_out4[1], tensor_to_insert4, rtol=1e-4, atol=1e-4)}")


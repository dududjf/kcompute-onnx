from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_sequence_construct import SequenceConstructOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

seq_construct_op = SequenceConstructOp(mgr)


def onnx_sequence_construct(*data) -> list:
    """ONNX reference implementation of SequenceConstruct"""
    return list(data)


# Case 1: Construct sequence from multiple tensors with different shapes
print("\nCase 1: Construct sequence from 3 tensors with different shapes")

# Create input tensors with different shapes
x1 = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
x2 = np.random.uniform(-1, 1, (5, 6)).astype(np.float32)
x3 = np.random.uniform(-1, 1, (7,)).astype(np.float32)

print(f"Input tensor 1 shape: {x1.shape}")
print(f"Input tensor 2 shape: {x2.shape}")
print(f"Input tensor 3 shape: {x3.shape}")

# NumPy reference
t0 = time.time()
numpy_out = onnx_sequence_construct(x1, x2, x3)
print(f"NumPy: {time.time() - t0} seconds")

# Kompute implementation
t1 = time.time()
kp_out = seq_construct_op.run(x1, x2, x3)
print(f"{seq_construct_op}: {time.time() - t1} seconds")

# Verify results
print(f"Number of tensors in sequence: NumPy={len(numpy_out)}, Kompute={len(kp_out)}")
print(f"Output tensor 1 shape: NumPy={numpy_out[0].shape}, Kompute={kp_out[0].shape}")
print(f"Output tensor 2 shape: NumPy={numpy_out[1].shape}, Kompute={kp_out[1].shape}")
print(f"Output tensor 3 shape: NumPy={numpy_out[2].shape}, Kompute={kp_out[2].shape}")

# Check if all tensors match
all_close = True
for i, (numpy_tensor, kp_tensor) in enumerate(zip(numpy_out, kp_out)):
    is_close = np.allclose(numpy_tensor, kp_tensor, rtol=1e-4, atol=1e-4)
    print(f"Tensor {i+1} all close: {is_close}")
    all_close = all_close and is_close

print(f"All tensors match: {all_close}")


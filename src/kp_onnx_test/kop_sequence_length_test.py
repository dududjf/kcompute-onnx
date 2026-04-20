from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_sequence_length import SequenceLengthOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

seq_length_op = SequenceLengthOp(mgr)


def onnx_sequence_length_reference(input_sequence):
    """ONNX reference implementation of SequenceLength"""
    if not isinstance(input_sequence, list):
        raise TypeError(
            f"input_sequence must be a list not {type(input_sequence)}."
        )
    return np.array(len(input_sequence), dtype=np.int64)


# Case 1: Get length of a sequence with 4 tensors
print("\nCase 1: Get length of a sequence with 4 tensors of different shapes")

# Create a sequence with 4 tensors
x1 = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
x2 = np.random.uniform(-1, 1, (4, 5, 6)).astype(np.float32)
x3 = np.random.uniform(-1, 1, (7,)).astype(np.float32)
x4 = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
sequence = [x1, x2, x3, x4]

print(f"Sequence contains {len(sequence)} tensors")
print(f"Tensor shapes: {[x.shape for x in sequence]}")

# NumPy reference
t0 = time.time()
numpy_out = onnx_sequence_length_reference(sequence)
print(f"NumPy: {time.time() - t0} seconds")

# Kompute implementation
t1 = time.time()
kp_out = seq_length_op.run(sequence)[0]
print(f"{seq_length_op}: {time.time() - t1} seconds")

# Verify results
print(f"\nSequence length: NumPy={numpy_out}, Kompute={int(kp_out[0])}")
print(f"Output shape: NumPy={numpy_out.shape}, Kompute={kp_out.shape}")
print(f"Output dtype: NumPy={numpy_out.dtype}, Kompute={kp_out.dtype}")

# Check if values match
length_match = int(numpy_out) == int(kp_out[0])
print(f"Length match: {length_match}")


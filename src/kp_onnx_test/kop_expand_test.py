from kp import Manager
import numpy as np
import time
from kp_onnx.kop_expand import ExpandOp


def common_reference_implementation(data: np.ndarray, shape: np.ndarray) -> np.ndarray:
    ones = np.ones(shape, dtype=data.dtype)
    return data * ones  # type: ignore

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
expand_op = ExpandOp(mgr)

print('Case 1')
numpy_in = np.arange(4, dtype=np.float32)
shape = np.array([3, 4], dtype=np.int32)

start_time = time.time()
numpy_out = common_reference_implementation(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = expand_op.run(numpy_in, shape)[0]
print(f"{expand_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 2')
numpy_in = np.random.random((2, 1)).astype(np.float32)
shape = np.array([2, 3], dtype=np.int32)

start_time = time.time()
numpy_out = common_reference_implementation(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = expand_op.run(numpy_in, shape)[0]
print(f"{expand_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 3')
numpy_in = np.random.random((3, 1, 5)).astype(np.float32)
shape = np.array([3, 4, 5], dtype=np.int32)

start_time = time.time()
numpy_out = common_reference_implementation(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = expand_op.run(numpy_in, shape)[0]
print(f"{expand_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 4')
numpy_in = np.random.random((3, 4, 5)).astype(np.float32)
shape = np.array([3, 1, 5], dtype=np.int32)

start_time = time.time()
numpy_out = common_reference_implementation(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = expand_op.run(numpy_in, shape)[0]
print(f"{expand_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 5')
numpy_in = np.arange(4, dtype=np.float32)
shape = np.array([0, 4], dtype=np.int32)

start_time = time.time()
numpy_out = common_reference_implementation(numpy_in, shape)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = expand_op.run(numpy_in, shape)[0]
print(f"{expand_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

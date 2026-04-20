from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_log_softmax import LogSoftmaxOp


# Reference implementation matching ONNX spec
def numpy_log_softmax(data: np.ndarray, axis: int = -1) -> np.ndarray:
    tmp = data - data.max(axis=axis, keepdims=1)  # type: ignore
    Y = np.exp(tmp)
    Y /= Y.sum(axis=axis, keepdims=1)  # type: ignore
    np.log(Y, out=Y)
    return Y


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

log_softmax_op = LogSoftmaxOp(mgr)

# Case 1: 3D tensor, axis: None
print("Case 1: 3D Tensor (32, 64, 256), axis: None")
x = np.random.uniform(-10.0, 10.0, (32, 64, 256)).astype(np.float32)
axis = -1

# NumPy baseline
start_time = time.time()
np_out = numpy_log_softmax(x, axis=axis)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute implementation
start_time = time.time()
log_softmax_op.axis = -1
kp_out = log_softmax_op.run(x)[0]
print(f"{log_softmax_op} time: {time.time() - start_time} seconds")

# Validation
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 2: 3D tensor, axis: 1
print("Case 2: 3D Tensor (32, 64, 256), axis: 1")
x = np.random.uniform(-10.0, 10.0, (32, 64, 256)).astype(np.float32)
axis = 1

# NumPy baseline
start_time = time.time()
np_out = numpy_log_softmax(x, axis=axis)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute implementation
start_time = time.time()
log_softmax_op.axis = 1
kp_out = log_softmax_op.run(x)[0]
print(f"{log_softmax_op} time: {time.time() - start_time} seconds")

# Validation
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 3: 3D tensor, axis: -2
print("Case 3: 3D Tensor (32, 64, 128), axis: -2")
x = np.random.uniform(-5.0, 5.0, (32, 64, 128)).astype(np.float32)
axis = -2

# NumPy baseline
start_time = time.time()
np_out = numpy_log_softmax(x, axis=axis)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute implementation
start_time = time.time()
log_softmax_op.axis = -2
kp_out = log_softmax_op.run(x)[0]
print(f"{log_softmax_op} time: {time.time() - start_time} seconds")

# Validation
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 4: 1D tensor, axis: 0
print("Case 4: 1D Tensor (128,), axis: 0")
x = np.random.uniform(-5.0, 5.0, (128,)).astype(np.float32)
axis = 0

# NumPy baseline
start_time = time.time()
np_out = numpy_log_softmax(x, axis=axis)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute implementation
start_time = time.time()
log_softmax_op.axis = 0
kp_out = log_softmax_op.run(x)[0]
print(f"{log_softmax_op} time: {time.time() - start_time} seconds")

# Validation
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

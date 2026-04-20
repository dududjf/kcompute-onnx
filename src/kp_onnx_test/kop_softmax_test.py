from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_softmax import SoftmaxOp


def onnx_softmax(X: np.ndarray, axis: int = -1) -> np.ndarray:
    tmp = X - X.max(axis=axis, keepdims=True)
    Y = np.exp(tmp)
    Y /= Y.sum(axis=axis, keepdims=True)
    return Y.astype(X.dtype, copy=False)

DEFAULT_AXIS = -1

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

softmax_op = SoftmaxOp(mgr)


# Case 1: 2D 默认（默认axis=-1）
print("Case 1: 2D, default axis")
x = np.random.uniform(-3, 3, (1024, 4096)).astype(np.float32)

t0 = time.time()
np_out = onnx_softmax(x, axis=DEFAULT_AXIS)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
softmax_op.axis = DEFAULT_AXIS
kp_out = softmax_op.run(x)[0]
print(f"{softmax_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 2: 2D，axis=0
print("Case 2: 2D, axis=0")
x = np.random.uniform(-4, 4, (4096, 1024)).astype(np.float32)

t0 = time.time()
np_out = onnx_softmax(x, axis=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
softmax_op.axis = 0
kp_out = softmax_op.run(x)[0]
print(f"{softmax_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 3: 3D 默认
print("Case 3: 3D, default axis")
x = np.random.uniform(-2, 2, (32, 64, 128)).astype(np.float32)

t0 = time.time()
np_out = onnx_softmax(x, axis=DEFAULT_AXIS)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
softmax_op.axis = DEFAULT_AXIS
kp_out = softmax_op.run(x)[0]
print(f"{softmax_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 4: 3D，axis=0
print("Case 4: 3D, axis=0")
x = np.random.uniform(-2, 2, (32, 64, 128)).astype(np.float32)

t0 = time.time()
np_out = onnx_softmax(x, axis=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
softmax_op.axis = 0
kp_out = softmax_op.run(x)[0]
print(f"{softmax_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 5: 3D，axis=2
print("Case 5: 3D, axis=2")
x = np.random.uniform(-2, 2, (32, 64, 128)).astype(np.float32)

t0 = time.time()
np_out = onnx_softmax(x, axis=2)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
softmax_op.axis = 2
kp_out = softmax_op.run(x)[0]
print(f"{softmax_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print("All close:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
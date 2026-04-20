from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_hardmax import HardmaxOp


def numpy_hardmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_argmax = np.argmax(x, axis=axis)
    y = np.zeros_like(x)
    np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
    return y


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

hardmax_op = HardmaxOp(mgr)

# Case 1
print("Case 1: 3D (32, 64, 512), axis=None")
x = np.random.uniform(-5, 5, (32, 64, 512)).astype(np.float32)
axis = -1

start_time = time.time()
np_out = numpy_hardmax(x, axis=axis)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
hardmax_op.axis = axis
kp_out = hardmax_op.run(x)[0]
print(f"{hardmax_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 2
print("Case 2: 3D (128, 64, 32), axis: 1")
x = np.random.uniform(-3, 3, (128, 64, 32)).astype(np.float32)
axis = 1

start_time = time.time()
np_out = numpy_hardmax(x, axis=axis)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
hardmax_op.axis = axis
kp_out = hardmax_op.run(x)[0]

print(f"{hardmax_op}: ", time.time() - start_time, "seconds")
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 3
print("Case 3: 3D (128, 64, 32), axis: -2")
x = np.random.uniform(-3, 3, (128, 64, 32)).astype(np.float32)
axis = -2

start_time = time.time()
np_out = numpy_hardmax(x, axis=axis)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
hardmax_op.axis = axis
kp_out = hardmax_op.run(x)[0]
print(f"{hardmax_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 4
print("Case 4: 1D (1024,), axis: 0")
x = np.random.uniform(-3, 3, (3, 64, 512, 32)).astype(np.float32)
axis = 0

start_time = time.time()
np_out = numpy_hardmax(x, axis)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
hardmax_op.axis = axis
kp_out = hardmax_op.run(x)[0]

print(f"{hardmax_op}: ", time.time() - start_time, "seconds")
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
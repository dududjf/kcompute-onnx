from kp import Manager
import numpy as np
import time
from kp_onnx.kop_lp_normalization import LpNormalizationOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

lp_normalization_op = LpNormalizationOp(mgr)


def np_lp_norm(x, axis=-1, p=2):
    norm = np.power(np.power(x, p).sum(axis=axis), 1.0 / p)
    norm = np.expand_dims(norm, axis)
    return (x / norm).astype(x.dtype)


x = np.random.random((255, 511, 1023)).astype(np.float32)

# -------- Case 1: data: 3D --------
print("Case 1: data: 3D")
start_time = time.time()
np_out = np_lp_norm(x)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = lp_normalization_op.run(x)[0]
print(f"{lp_normalization_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: data: 3D, axis: 0 --------
print("Case 2: data: 3D, axis: -2")

start_time = time.time()
np_out = np_lp_norm(x, -2)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
lp_normalization_op.axis = -2
kp_out = lp_normalization_op.run(x)[0]
print(f"{lp_normalization_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: data: 3D, axis=-2, p: 1 --------
print("Case 3: data: 3D, axis: 0, p: 1")

start_time = time.time()
np_out = np_lp_norm(x, 0, 1)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
lp_normalization_op.axis = 0
lp_normalization_op.p = 1
kp_out = lp_normalization_op.run(x)[0]
print(f"{lp_normalization_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

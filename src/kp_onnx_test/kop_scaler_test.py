from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_scaler import ScalerOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())


def onnx_scaler(x, offset=None, scale=None):
    dx = x - offset
    return (dx * scale).astype(x.dtype)


# -------- Case 1: 标量 offset 和 scale --------
print("Case 1: 标量 offset 和 scale")
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)
offset = [1.0]
scale = [2.0]

start_time = time.time()
np_out = onnx_scaler(x, offset=offset, scale=scale)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
scaler_op = ScalerOp(mgr, offset=offset, scale=scale)
kp_out = scaler_op.run(x)[0]
print(f"{scaler_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: 向量 offset 和 scale--------
print("Case 2: 向量 offset 和 scale")
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)
offset = [0.0, 1.0, 2.0]
scale = [1.0, 2.0, 3.0]

start_time = time.time()
np_out = onnx_scaler(x, offset=offset, scale=scale)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
scaler_op = ScalerOp(mgr, offset=offset, scale=scale)
kp_out = scaler_op.run(x)[0]
print(f"{scaler_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: 3D --------
print("Case 3: 3D")
x = np.random.randn(32, 512, 2).astype(np.float32)
offset = [0.5, 3.0]
scale = [2.1, 4.0]

start_time = time.time()
np_out = onnx_scaler(x, offset=offset, scale=scale)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
scaler_op = ScalerOp(mgr, offset=offset, scale=scale)
kp_out = scaler_op.run(x)[0]
print(f"{scaler_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

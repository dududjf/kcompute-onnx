from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_global_average_pool import GlobalAveragePoolOp

# ONNX的GAP实现
def onnx_gap(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, x.ndim))
    y = np.average(x, axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y.astype(x.dtype, copy=False)

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

gap_op = GlobalAveragePoolOp(mgr)

# Case 1: 2D 图像场景 NCHW
print("Case 1: NCHW (2D)")
numpy_in = np.random.uniform(-3, 3, (4, 16, 128, 256))

start_time = time.time()
numpy_out = onnx_gap(numpy_in)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gap_op.run(numpy_in)[0]
print(f"{gap_op}:", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 1D 序列场景 NCL
print("Case 2: NCL (1D)")
numpy_in = np.random.uniform(-2, 2, (8, 8, 1024))

start_time = time.time()
numpy_out = onnx_gap(numpy_in)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gap_op.run(numpy_in)[0]
print(f"{gap_op}:", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 3D 数据场景 NCDHW
print("Case 3: NCDHW (3D)")
numpy_in = np.random.uniform(-1, 1, (2, 4, 16, 32, 8))

start_time = time.time()
numpy_out = onnx_gap(numpy_in)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gap_op.run(numpy_in)[0]
print(f"{gap_op}:", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 无空间维，只有N、C
print("Case 4: NC Only")
numpy_in = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)

start_time = time.time()
axes = tuple(range(2, numpy_in.ndim))
numpy_out = onnx_gap(numpy_in)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = gap_op.run(numpy_in)[0]
print(f"{gap_op}:", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
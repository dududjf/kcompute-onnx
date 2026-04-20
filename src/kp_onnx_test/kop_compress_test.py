import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_compress import CompressOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

compress_op = CompressOp(mgr)

print("Case 1")
x = np.random.random((3, 2, 51, 511)).astype(np.float32)
condition = np.array([1, 1, 0, 0], dtype=np.int64)

start_time = time.time()
np_out = np.compress(condition, x)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
compress_op.axis = None
kp_out = compress_op.run(x, condition)[0]
print(f"{compress_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 2")
x = np.random.random((3, 3, 3, 3)).astype(np.float32)
condition = np.random.random_integers(0, 1, (81,)).astype(np.int64)

start_time = time.time()
np_out = np.compress(condition, x)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
compress_op.axis = None
kp_out = compress_op.run(x, condition)[0]
print(f"{compress_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 3")
x = np.random.random((3, 2, 3, 3)).astype(np.float32)
condition = np.random.random_integers(0, 1, (3,)).astype(np.int64)
axis = -1

start_time = time.time()
np_out = np.compress(condition, x, axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
compress_op.axis = axis
kp_out = compress_op.run(x, condition)[0]
print(f"{compress_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 4")
x = np.random.random((3, 20, 3, 3)).astype(np.float32)
condition = np.random.random_integers(0, 1, (20,)).astype(np.int64)
axis = 1

start_time = time.time()
np_out = np.compress(condition, x, axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
compress_op.axis = axis
kp_out = compress_op.run(x, condition)[0]
print(f"{compress_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')
from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_celu import CeluOp

DEFAULT_ALPHA = 1.0

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

celu_op = CeluOp(mgr)


# Case 1: 默认(alpha=1.0)
numpy_in1 = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(numpy_in1 >= 0.0, numpy_in1, DEFAULT_ALPHA * (np.exp(numpy_in1 / DEFAULT_ALPHA) - 1.0))
print("NumPy (Default):", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = celu_op.run(numpy_in1)[0]
print(f"{celu_op} (Default):", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 2: 指定 alpha
numpy_in2 = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

start_time = time.time()
alpha = 0.5
numpy_out = np.where(numpy_in2 >= 0.0, numpy_in2, alpha * (np.exp(numpy_in2 / alpha) - 1.0))
print("NumPy (alpha=0.5):", time.time() - start_time, "seconds")

start_time = time.time()
celu_op.alpha = alpha
kp_out = celu_op.run(numpy_in2)[0]
print(f"{celu_op} (alpha=0.5):", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

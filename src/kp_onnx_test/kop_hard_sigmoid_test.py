from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_hard_sigmoid import HardSigmoidOp

DEFAULT_ALPHA = float(0.2)
DEFAULT_BETA  = float(0.5)

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

hard_sigmoid_op = HardSigmoidOp(mgr)


# Case 1: 单一输入参数(默认alpha = 0.2、beta = 0.5)
x1 = np.random.uniform(-8, 8, (10240, 40960)).astype(np.float32)

start_time = time.time()
numpy_out = np.clip(DEFAULT_ALPHA * x1 + DEFAULT_BETA, 0.0, 1.0)
print("NumPy [default alpha,beta]:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = hard_sigmoid_op.run(x1)[0]
print(f"{hard_sigmoid_op} [default alpha,beta]:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 2: 指定 alpha
x2 = np.random.uniform(-8, 8, (10240, 40960)).astype(np.float32)
alpha = np.float32(0.35)

start_time = time.time()
numpy_out = np.clip(alpha * x2 + DEFAULT_BETA, 0.0, 1.0)
print("NumPy [alpha=0.35, beta=default]:", time.time() - start_time, "seconds")

start_time = time.time()
hard_sigmoid_op.alpha = alpha
kp_out = hard_sigmoid_op.run(x2)[0]
print(f"{hard_sigmoid_op} [alpha=0.35, beta=default]:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 3: 同时指定 alpha、beta
x3 = np.random.uniform(-8, 8, (10240, 40960)).astype(np.float32)
alpha = np.float32(0.1)
beta  = np.float32(0.6)

start_time = time.time()
numpy_out = np.clip(alpha * x3 + beta, 0.0, 1.0)
print("NumPy [alpha=0.1, beta=0.6]:", time.time() - start_time, "seconds")

start_time = time.time()
hard_sigmoid_op.alpha = alpha
hard_sigmoid_op.beta  = beta
kp_out = hard_sigmoid_op.run(x3)[0]
print(f"{hard_sigmoid_op} [alpha=0.1, beta=0.6]:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
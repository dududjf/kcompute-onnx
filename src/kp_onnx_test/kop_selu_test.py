from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_selu import SeluOp

DEFAULT_ALPHA = float(1.6732632423543772)
DEFAULT_GAMMA = float(1.0507009873554805)

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

selu_op = SeluOp(mgr)


# Case 1: 单一输入参数（默认alpha=1.6732632423543772、gamma=1.0507009873554805）
x1 = np.random.uniform(-5, 5, (10240, 40960)).astype(np.float32)

start_time = time.time()
numpy_out = DEFAULT_GAMMA * np.where(x1 >= 0.0, x1, DEFAULT_ALPHA * (np.exp(x1) - 1.0))
print("NumPy [default alpha,gamma]:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = selu_op.run(x1)[0]
print(f"{selu_op} [default alpha,gamma]:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 2: 指定 alpha
x2 = np.random.uniform(-5, 5, (10240, 40960)).astype(np.float32)
alpha = np.float32(0.8)

start_time = time.time()
numpy_out = DEFAULT_GAMMA * np.where(x2 >= 0.0, x2, alpha * (np.exp(x2) - 1.0))
print("NumPy [alpha=0.8, gamma=default]:", time.time() - start_time, "seconds")

start_time = time.time()
selu_op.alpha = alpha
kp_out = selu_op.run(x2)[0]
print(f"{selu_op} [alpha=0.8, gamma=default]:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 3: 同时指定 alpha、gamma
x3 = np.random.uniform(-5, 5, (10240, 40960)).astype(np.float32)
alpha = np.float32(1.2)
gamma = np.float32(0.9)

start_time = time.time()
numpy_out = gamma * np.where(x3 >= 0.0, x3, alpha * (np.exp(x3) - 1.0))
print("NumPy [alpha=1.2, gamma=0.9]:", time.time() - start_time, "seconds")

start_time = time.time()
selu_op.alpha = alpha
selu_op.gamma = gamma
kp_out = selu_op.run(x3)[0]
print(f"{selu_op} [alpha=1.2, gamma=0.9]:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
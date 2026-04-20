from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_leaky_relu import LeakyReluOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

leaky_relu_op = LeakyReluOp(mgr)

# ---------------- Case 1: alpha: None ----------------
print("Case 1: LeakyReLU alpha: None")
x = np.random.random((1024, 1024)).astype(np.float32)
alpha = 0.01

start_time = time.time()
np_out = np.where(x >= 0.0, x, alpha * x)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
leaky_relu_op.alpha = alpha
kp_out = leaky_relu_op.run(x)[0]
print(f"{leaky_relu_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 2: alpha: 0.2 ----------------
print("Case 2: LeakyReLU alpha=0.2")
x = np.random.random((1024, 1024))
alpha = 0.2

start_time = time.time()
np_out = np.where(x >= 0.0, x, alpha * x)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
leaky_relu_op.alpha = alpha
kp_out = leaky_relu_op.run(x)[0]
print(f"{leaky_relu_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")



from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_elu import EluOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

elu_op = EluOp(mgr)

# ---------------- Case 1: alpha: None ----------------
print("Case 1: ELU alpha: None")
x = np.random.random((1024, 1024)).astype(np.float32)
print("Input shape:", x.shape)

# Numpy
start_time = time.time()
np_out = np.where(x > 0, x, 1.0 * (np.exp(x) - 1.0))
print("Numpy: ", time.time() - start_time, "seconds")

# Kompute
start_time = time.time()
elu_op.alpha = 1.0
kp_out = elu_op.run(x)[0]
print(f"{elu_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 2: alpha=0.5 ----------------
print("Case 2: ELU alpha=0.5")
x = np.random.random((1024, 1024))
alpha = 0.5
print("Input shape:", x.shape)

# Numpy
start_time = time.time()
np_out = np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
print("Numpy: ", time.time() - start_time, "seconds")

# Kompute
start_time = time.time()
elu_op.alpha = alpha
kp_out = elu_op.run(x)[0]
print(f"{elu_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")



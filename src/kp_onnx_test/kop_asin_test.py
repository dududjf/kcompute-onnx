from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_asin import AsinOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

asin_op = AsinOp(mgr)

x = np.random.uniform(-1.0, 1.0, (4096, 4096)).astype(np.float32)

# NumPy
start_time = time.time()
np_out = np.arcsin(x)
print("Numpy:", time.time() - start_time, "seconds")

# AsinOp
start_time = time.time()
kp_out = asin_op.run(x)[0]
print(f"{asin_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

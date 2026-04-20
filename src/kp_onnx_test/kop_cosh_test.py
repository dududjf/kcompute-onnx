from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_cosh import CoshOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

cosh_op = CoshOp(mgr)

x = np.random.uniform(-5.0, 5.0, (4096, 4096)).astype(np.float32)

# NumPy baseline
start_time = time.time()
np_out = np.cosh(x)
print("Numpy:", time.time() - start_time, "seconds")

# CoshOp
start_time = time.time()
kp_out = cosh_op.run(x)[0]
print(f"{cosh_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

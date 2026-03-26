from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_softplus import SoftplusOp

# Device info
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

softplus_op = SoftplusOp(mgr)

x = np.random.uniform(-20.0, 20.0, (4096, 4096)).astype(np.float32)

# NumPy baseline
start_time = time.time()
np_out = np.log(np.exp(x).astype(x.dtype) + 1)
print("Numpy:", time.time() - start_time, "seconds")

# SoftplusOp
start_time = time.time()
kp_out = softplus_op.run(x)[0]
print(f"{softplus_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

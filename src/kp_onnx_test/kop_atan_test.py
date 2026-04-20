from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_atan import AtanOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

atan_op = AtanOp(mgr)

numpy_in = np.random.uniform(-1000, 1000, 1024 * 1024 * 16).astype(np.float32)

start_time = time.time()
numpy_out = np.arctan(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = atan_op.run(numpy_in)[0]
print(f"{atan_op}:", time.time() - start_time, "seconds")

print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

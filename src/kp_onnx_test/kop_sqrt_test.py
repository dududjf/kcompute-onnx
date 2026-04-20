from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_sqrt import SqrtOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sqrt_op = SqrtOp(mgr)

numpy_in = np.random.random(1024 * 1024 * 16)

start_time = time.time()
numpy_out = np.sqrt(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sqrt_op.run(numpy_in)[0]
print(f"{sqrt_op}:", time.time() - start_time, "seconds")

print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

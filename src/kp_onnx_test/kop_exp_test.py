from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_exp import ExpOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

exp_op = ExpOp(mgr)

numpy_in = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

start_time = time.time()
numpy_out = np.exp(numpy_in)
print("EXP Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = exp_op.run(numpy_in)[0]
print(f"{exp_op}:", time.time() - start_time, "seconds")

print("EXP Max error:", np.abs(numpy_out - kp_out).max())
print("EXP All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

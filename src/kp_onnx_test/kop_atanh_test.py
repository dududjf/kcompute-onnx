from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_atanh import AtanhOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

atanh_op = AtanhOp(mgr)

# atanh 的定义域是 (-1, 1)，避免靠近极点带来无穷/NaN
numpy_in = np.random.uniform(-0.99, 0.99, (640, 10240))

start_time = time.time()
numpy_out = np.arctanh(numpy_in)
print("ATANH Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = atanh_op.run(numpy_in)[0]
print(f"{atanh_op}:", time.time() - start_time, "seconds")

print("ATANH Max error:", np.abs(numpy_out - kp_out).max())
print("ATANH All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

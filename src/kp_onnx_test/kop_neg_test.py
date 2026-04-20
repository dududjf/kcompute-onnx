from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_neg import NegOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

neg_op = NegOp(mgr)

numpy_in = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

start_time = time.time()
numpy_out = -numpy_in
print("NEG Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = neg_op.run(numpy_in)[0]
print(f"{neg_op}:", time.time() - start_time, "seconds")

print('NEG Max error:', np.abs(numpy_out - kp_out).max())
print("NEG All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

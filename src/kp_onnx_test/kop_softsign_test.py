from kp import Manager
import numpy as np
import time
from kp_onnx.kop_softsign import SoftsignOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

softsign_op = SoftsignOp(mgr)

numpy_in = np.random.uniform(-10, 10, (10240, 40960)).astype(np.float32)

start_time = time.time()
numpy_out = numpy_in / (1.0 + np.abs(numpy_in))
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = softsign_op.run(numpy_in)[0]
print(f"{softsign_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
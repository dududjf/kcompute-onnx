from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_tan import TanOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

tan_op = TanOp(mgr)

numpy_in = np.random.uniform(-1.2, 1.2, (10240, 40960)).astype(np.float32)

start_time = time.time()
numpy_out = np.tan(numpy_in)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = tan_op.run(numpy_in)[0]
print(f"{tan_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
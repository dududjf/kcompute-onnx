import numpy as np
import time
from kp import Manager
from kp_onnx_ssbo.kop_log import LogOp

device_id = 1
mgr = Manager(device_id)
print(mgr.get_device_properties())

log_op = LogOp(mgr)

x = np.random.random((1024, 1024)).astype(np.float32)
print("input shape:", x.shape)
start_time = time.time()
numpy_out = np.log(x)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kop_out = log_op.run(x)[0]
print(f"{log_op}: ", time.time() - start_time, "seconds")

# print(numpy_out)
# print(kop_out)
print("Max error:", np.max(np.abs(numpy_out - kop_out)))
print(np.allclose(numpy_out, kop_out, rtol=1e-4, atol=1e-4))

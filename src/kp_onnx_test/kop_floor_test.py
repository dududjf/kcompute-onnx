from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_floor import FloorOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

floor_op = FloorOp(mgr)

x = np.random.random((10240, 10240)).astype(np.float32)
start_time = time.time()
np_out = np.floor(x)
print("Numpy: ", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = floor_op.run(x)[0]
print(f"{floor_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

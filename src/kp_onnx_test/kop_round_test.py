from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_round import RoundOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

round_op = RoundOp(mgr)

x = np.random.uniform(-5, 5, (1024, 1024)).astype(np.float32)

# Numpy baseline (np.round default: ties-to-even)
start_time = time.time()
np_out = np.round(x)
print("Numpy:", time.time() - start_time, "seconds")

# RoundOp
start_time = time.time()
kp_out = round_op.run(x)[0]
print(f"{round_op}: ", time.time() - start_time, "seconds")

print("Max error: ", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


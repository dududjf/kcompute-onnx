from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_relu import ReluOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

relu_op = ReluOp(mgr)

numpy_in = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

start_time = time.time()
numpy_out = np.maximum(numpy_in, 0.0)
print("RELU Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = relu_op.run(numpy_in)[0]
print(f"{relu_op}:", time.time() - start_time, "seconds")

print("RELU Max error:", np.abs(numpy_out - kp_out).max())
print("RELU All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

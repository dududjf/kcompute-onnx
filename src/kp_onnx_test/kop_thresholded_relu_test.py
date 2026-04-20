from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_thresholded_relu import ThresholdedReluOp, DEFAULT_ALPHA

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

thresholded_relu_op = ThresholdedReluOp(mgr)


# Case 1: 单一输入参数
numpy_in1 = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(numpy_in1 > DEFAULT_ALPHA, numpy_in1, 0.0).astype(np.float32)
print("NumPy (Default alpha=1.0):", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = thresholded_relu_op.run(numpy_in1)[0]
print(f"{thresholded_relu_op} (Default alpha=1.0):", time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 2: 设置 alpha = 0.5
numpy_in2 = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)

alpha = 0.5
start_time = time.time()
numpy_out = np.where(numpy_in2 > alpha, numpy_in2, 0.0).astype(np.float32)
print("NumPy (alpha=0.5):", time.time() - start_time, "seconds")

start_time = time.time()
thresholded_relu_op.alpha = alpha
kp_out = thresholded_relu_op.run(numpy_in2)[0]
print(f"{thresholded_relu_op} (alpha=0.5):", time.time() - start_time, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
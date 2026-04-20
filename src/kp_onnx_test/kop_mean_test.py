from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_mean import MeanOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
mean_op = MeanOp(mgr)

# Case 1: 单输入
print('Case 1 (single input)')
numpy_in = np.random.random((640, 1024))

start_time = time.time()
numpy_out = numpy_in.copy()
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 2: 双输入，低维 → 高维
print('Case 2')
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((3, 1023, 1023, 1))

start_time = time.time()
numpy_out = (numpy_in_1 + numpy_in_2) / 2.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 3: 双输入，交换顺序
print('Case 3')
numpy_in_1 = np.random.random((3, 1023, 1023, 1))
numpy_in_2 = np.random.random((1023, 15))

start_time = time.time()
numpy_out = (numpy_in_1 + numpy_in_2) / 2.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 4: 双输入，更多高维广播
print('Case 4')
numpy_in_1 = np.random.random((255, 255, 15))
numpy_in_2 = np.random.random((3, 3, 255, 255, 1))

start_time = time.time()
numpy_out = (numpy_in_1 + numpy_in_2) / 2.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 5: 双输入，交换顺序
print('Case 5')
numpy_in_1 = np.random.random((3, 3, 255, 255, 1))
numpy_in_2 = np.random.random((255, 255, 15))

start_time = time.time()
numpy_out = (numpy_in_1 + numpy_in_2) / 2.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 6: 双输入，按列广播
print('Case 6')
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((1023, 1))

start_time = time.time()
numpy_out = (numpy_in_1 + numpy_in_2) / 2.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 7: 双输入，标量广播
print('Case 7')
numpy_in_1 = np.random.random((1023,))
numpy_in_2 = np.random.random((1,))

start_time = time.time()
numpy_out = (numpy_in_1 + numpy_in_2) / 2.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 8: 三输入混合广播，包含标量（1 元张量）→ 向量
print('Case 8')
a = np.random.random((1023, 15))
b = np.random.random((1,))
c = np.random.random((1023, 1))

start_time = time.time()
numpy_out = (a + b + c) / 3.0
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mean_op.run(a, b, c)[0]
print(f"{mean_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

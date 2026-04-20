from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_sum import SumOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
sum_op = SumOp(mgr)

print('Case 1')
numpy_in = np.random.random((640, 1024))

start_time = time.time()
numpy_out = numpy_in.copy()
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 2')
numpy_in_1 = np.random.random((3, 1023, 1023, 1))
numpy_in_2 = np.random.random((1023, 15))

start_time = time.time()
numpy_out = numpy_in_1 + numpy_in_2
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 3')
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((3, 1023, 1023, 1))

start_time = time.time()
numpy_out = numpy_in_1 + numpy_in_2
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 4')
numpy_in_1 = np.random.random((255, 255, 15))
numpy_in_2 = np.random.random((3, 3, 255, 255, 1))

start_time = time.time()
numpy_out = numpy_in_1 + numpy_in_2
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 5')
numpy_in_1 = np.random.random((3, 3, 255, 255, 1))
numpy_in_2 = np.random.random((255, 255, 15))

start_time = time.time()
numpy_out = numpy_in_1 + numpy_in_2
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 6')
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((1023, 1))

start_time = time.time()
numpy_out = numpy_in_1 + numpy_in_2
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 7')
numpy_in_1 = np.random.random((1023,))
numpy_in_2 = np.random.random((1,))

start_time = time.time()
numpy_out = numpy_in_1 + numpy_in_2
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sum_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 8')
def numpy_sum(*data):
    if len(data) == 1:
        return data[0].copy()
    if len(data) == 2:
        return data[0] + data[1]
    a = data[0]
    for i in range(1, len(data)):
        a = a + data[i]
    return a

a = np.random.random((1023, 15))
b = np.random.random((1,))
c = np.random.random((1023, 1))

t0 = time.time()
numpy_out = numpy_sum(a, b, c)
print("NumPy:", numpy_out.shape, time.time() - t0, "seconds")

t1 = time.time()
kp_out = sum_op.run(a, b, c)[0]
print(f"{sum_op}:", kp_out.shape, time.time() - t1, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

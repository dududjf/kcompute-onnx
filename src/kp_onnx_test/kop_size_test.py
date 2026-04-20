from kp import Manager
import numpy as np
import time
from kp_onnx.kop_size import SizeOp


def size_reference(data: np.ndarray):
    return np.array(data.size, dtype=np.int64)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
size_op = SizeOp(mgr)

# Case 1: 标量 tensor
numpy_in = np.array(42)

start_time = time.time()
numpy_out = size_reference(numpy_in)
print("Numpy:", numpy_out, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = size_op.run(numpy_in)[0]
print(f"{size_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 2: 一维数组
numpy_in = np.random.random(5)

start_time = time.time()
numpy_out = size_reference(numpy_in)
print("Numpy:", numpy_out, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = size_op.run(numpy_in)[0]
print(f"{size_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 3: 二维数组
numpy_in = np.random.random((3, 4))

start_time = time.time()
numpy_out = size_reference(numpy_in)
print("Numpy:", numpy_out, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = size_op.run(numpy_in)[0]
print(f"{size_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 4: 三维数组
numpy_in = np.random.random((2, 3, 4))

start_time = time.time()
numpy_out = size_reference(numpy_in)
print("Numpy:", numpy_out, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = size_op.run(numpy_in)[0]
print(f"{size_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

# Case 5: 多维数组
numpy_in = np.random.random((2, 3, 4, 5))

start_time = time.time()
numpy_out = size_reference(numpy_in)
print("Numpy:", numpy_out, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = size_op.run(numpy_in)[0]
print(f"{size_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4), "\n")

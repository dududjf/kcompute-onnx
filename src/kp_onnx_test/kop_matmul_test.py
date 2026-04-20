from kp import Manager
import numpy as np
import time
from kp_onnx.kop_matmul import MatMulOp

device_id = 1
mgr = Manager(device_id)
print(mgr.get_device_properties())

matmul_op = MatMulOp(mgr)
numpy_in_1 = np.random.random((2, 5, 1000, 512))
numpy_in_2 = np.random.random((512, 1024))

start_time = time.time()
numpy_out = np.matmul(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

numpy_in_1 = np.random.random((2, 5, 1000, 512))
numpy_in_2 = np.random.random((2, 5, 512, 1024))

start_time = time.time()
numpy_out = np.matmul(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

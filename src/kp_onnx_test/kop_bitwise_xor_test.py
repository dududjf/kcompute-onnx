from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_bitwise_xor import BitwiseXorOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
xor_op = BitwiseXorOp(mgr)

print('Case 1')
numpy_in_1 = np.random.randint(0, 100, size=(1023, 15), dtype=np.int32)
numpy_in_2 = np.random.randint(0, 100, size=(3, 1023, 1023, 1), dtype=np.int32)

start_time = time.time()
numpy_out = np.bitwise_xor(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = xor_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{xor_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 2')
numpy_in_1 = np.random.randint(0, 100, size=(3, 1023, 1023, 1), dtype=np.int32)
numpy_in_2 = np.random.randint(0, 100, size=(1023, 15), dtype=np.int32)

start_time = time.time()
numpy_out = np.bitwise_xor(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = xor_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{xor_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 3')
numpy_in_1 = np.random.randint(0, 100, size=(255, 255, 15), dtype=np.int32)
numpy_in_2 = np.random.randint(0, 100, size=(3, 3, 255, 255, 1), dtype=np.int32)

start_time = time.time()
numpy_out = np.bitwise_xor(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = xor_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{xor_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 4')
numpy_in_1 = np.random.randint(0, 100, size=(3, 3, 255, 255, 1), dtype=np.int32)
numpy_in_2 = np.random.randint(0, 100, size=(255, 255, 15), dtype=np.int32)

start_time = time.time()
numpy_out = np.bitwise_xor(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = xor_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{xor_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 5')
numpy_in_1 = np.random.randint(0, 100, size=(1023, 15), dtype=np.int32)
numpy_in_2 = np.random.randint(0, 100, size=(1023, 1), dtype=np.int32)

start_time = time.time()
numpy_out = np.bitwise_xor(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = xor_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{xor_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 6')
numpy_in_1 = np.random.randint(0, 100, size=(1023,), dtype=np.int32)
numpy_in_2 = np.random.randint(0, 100, size=(1,), dtype=np.int32)

start_time = time.time()
numpy_out = np.bitwise_xor(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = xor_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{xor_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-5, atol=1e-5))

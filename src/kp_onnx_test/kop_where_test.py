from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_where import WhereOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
where_op = WhereOp(mgr)

print('Case 1')
condition = np.random.choice([0, 1], size=(1023, 1)).astype(np.float32)
x = np.random.random((1023, 15)).astype(np.float32)
y = np.random.random((1023, 1)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}: ", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 2')
condition = np.random.choice([0, 1], size=(1023, 1)).astype(np.float32)
x = np.random.random((1023, 15)).astype(np.float32)
y = np.random.random((3, 1023, 1023, 1)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 3')
condition = np.random.choice([0, 1], size=(1,)).astype(np.float32)
x = np.random.random((1,)).astype(np.float32)
y = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 4')
condition = np.random.choice([0, 1], size=(1023,)).astype(np.float32)
x = np.random.random((1023,)).astype(np.float32)
y = np.random.random((1,)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 5')
condition = np.random.choice([0, 1], size=(3, 3, 255, 255, 1)).astype(np.float32)
x = np.random.random((3, 3, 255, 255, 1)).astype(np.float32)
y = np.random.random((255, 255, 15)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 6')
condition = np.random.choice([0, 1], size=(3, 1, 255, 1)).astype(np.float32)
x = np.random.random((3, 3, 255, 255, 1)).astype(np.float32)
y = np.random.random((3, 1, 255, 15)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 7')
condition = np.random.choice([0, 1], size=(255, 1)).astype(np.float32)
x = np.random.random((225, 255, 15)).astype(np.float32)
y = np.random.random((3, 3, 225, 255, 1)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

print('Case 8')
condition = np.random.choice([0, 1], size=(1023, 1)).astype(np.float32)
x = np.random.random((3, 1023, 1023, 1)).astype(np.float32)
y = np.random.random((1023, 15)).astype(np.float32)

start_time = time.time()
numpy_out = np.where(condition, x, y)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = where_op.run(condition, x, y)[0]
print(f"{where_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
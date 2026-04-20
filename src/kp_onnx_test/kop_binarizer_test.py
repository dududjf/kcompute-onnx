from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_binarizer import BinarizerOp


def np_binarizer(x, threshold):
    return (x > threshold).astype(x.dtype)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
binarizer_op = BinarizerOp(mgr, threshold=0.5)

print('Case 1')
numpy_in = np.random.random((3, 255, 511, 15)).astype(np.float32)
start_time = time.time()
np_out = np_binarizer(numpy_in, threshold=0.5)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = binarizer_op.run(numpy_in)[0]
print(f"{binarizer_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 2')
numpy_in = np.random.random((3, 3, 255, 255, 15)).astype(np.float32)
start_time = time.time()
np_out = np_binarizer(numpy_in, threshold=0.5)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = binarizer_op.run(numpy_in)[0]
print(f"{binarizer_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 3')
numpy_in = np.random.random((3, 3, 255, 255, 15)).astype(np.float32)
threshold_in = np.random.random((1,)).astype(np.float32)
binarizer_op_with_threshold = BinarizerOp(mgr, threshold=threshold_in[0])

start_time = time.time()
np_out = np_binarizer(numpy_in, threshold=threshold_in[0])
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = binarizer_op_with_threshold.run(numpy_in)[0]
print(f"{binarizer_op_with_threshold}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 4')
numpy_in = np.random.random((1023, 15)).astype(np.float32)
start_time = time.time()
np_out = np_binarizer(numpy_in, threshold=0.5)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = binarizer_op.run(numpy_in)[0]
print(f"{binarizer_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 5')
numpy_in = np.random.random((1023,)).astype(np.float32)
threshold_in = np.random.random((1,)).astype(np.float32)
binarizer_op_with_threshold = BinarizerOp(mgr, threshold=threshold_in[0])

start_time = time.time()
np_out = np_binarizer(numpy_in, threshold=threshold_in[0])
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = binarizer_op_with_threshold.run(numpy_in)[0]
print(f"{binarizer_op_with_threshold}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

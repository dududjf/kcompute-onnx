from kp import Manager
import numpy as np
import time
from kp_onnx.kop_prelu import PReLUOp


def np_prelu(x, slope):
    try:
        return np.where(x > 0, x, x * slope).astype(x.dtype)
    except ValueError:
        if len(slope.shape) == 1:
            dim = slope.shape[0]
            new_shape = []
            n = 0
            for d in x.shape:
                if d == dim:
                    new_shape.append(d)
                    n += 1
                else:
                    new_shape.append(1)
            if n == 1:
                xs = x * slope.reshape(tuple(new_shape))
                return np.where(x > 0, x, xs).astype(x.dtype)
        raise


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
prelu_op = PReLUOp(mgr)

print('Case 1')
numpy_in = np.random.random((3, 255, 511, 15)).astype(np.float32)
slope_in = np.random.random((511, 1)).astype(np.float32)

start_time = time.time()
np_out = np_prelu(numpy_in, slope_in)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = prelu_op.run(numpy_in, slope_in)[0]
print(f"{prelu_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 2')
numpy_in = np.random.random((3, 3, 255, 255, 15)).astype(np.float32)
slope_in = np.random.random((255, 255, 1)).astype(np.float32)

start_time = time.time()
np_out = np_prelu(numpy_in, slope_in)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = prelu_op.run(numpy_in, slope_in)[0]
print(f"{prelu_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 3')
numpy_in = np.random.random((3, 3, 255, 255, 15)).astype(np.float32)
slope_in = np.random.random((3, 1, 255, 1)).astype(np.float32)

start_time = time.time()
np_out = np_prelu(numpy_in, slope_in)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = prelu_op.run(numpy_in, slope_in)[0]
print(f"{prelu_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 4')
numpy_in = np.random.random((1023, 15)).astype(np.float32)
slope_in = np.random.random((1023, 1)).astype(np.float32)

start_time = time.time()
np_out = np_prelu(numpy_in, slope_in)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = prelu_op.run(numpy_in, slope_in)[0]
print(f"{prelu_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 5')
numpy_in = np.random.random((1023, 15)).astype(np.float32)
slope_in = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
np_out = np_prelu(numpy_in, slope_in)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = prelu_op.run(numpy_in, slope_in)[0]
print(f"{prelu_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

print('Case 6')
numpy_in = np.random.random((1023,)).astype(np.float32)
slope_in = np.random.random((1,)).astype(np.float32)

start_time = time.time()
np_out = np_prelu(numpy_in, slope_in)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = prelu_op.run(numpy_in, slope_in)[0]
print(f"{prelu_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

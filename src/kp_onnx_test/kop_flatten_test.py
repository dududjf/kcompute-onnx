from kp import Manager
import numpy as np
import time
from kp_onnx.kop_flatten import FlattenOp


def np_flatten(x, axis=1):
    i = axis
    if i < 0:
        i += len(x.shape)
    if i == 0:
        new_shape = (1, -1)
    else:
        new_shape = (int(np.prod(x.shape[:i])), -1)
    return x.reshape(new_shape)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
flatten_op = FlattenOp(mgr)

# Case 1 : 4D tensor, axis=1
print('Case 1')
numpy_in = np.random.random((3, 1023, 255, 15)).astype(np.float32)
flatten_op.axis = 1

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 2 : 4D tensor, axis=2
print('Case 2')
numpy_in = np.random.random((3, 255, 1023, 15)).astype(np.float32)
flatten_op.axis = 2

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 3 : 5D tensor, axis=3
print('Case 3')
numpy_in = np.random.random((3, 3, 255, 255, 15)).astype(np.float32)
flatten_op.axis = 3

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 4 : 2D tensor, axis=1
print('Case 4')
numpy_in = np.random.random((1023, 255)).astype(np.float32)
flatten_op.axis = 1

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 5 : 2D tensor, axis=0
print('Case 5')
numpy_in = np.random.random((1023, 15)).astype(np.float32)
flatten_op.axis = 0

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 6 : 1D tensor, axis=0
print('Case 6')
numpy_in = np.random.random((1023,)).astype(np.float32)
flatten_op.axis = 0

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 7 : 3D tensor, negative axis=-1
print('Case 7')
numpy_in = np.random.random((3, 225, 255)).astype(np.float32)
flatten_op.axis = -1

start_time = time.time()
np_out = np_flatten(numpy_in, axis=flatten_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = flatten_op.run(numpy_in)[0]
print(f"{flatten_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

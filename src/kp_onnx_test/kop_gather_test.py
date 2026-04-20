import time
import numpy as np
from kp import Manager
from kp_onnx.kop_gather import GatherOp


def np_gather(x, indices, axis=0):
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not indices.flags["C_CONTIGUOUS"]:
        indices = np.ascontiguousarray(indices)
    if indices.size == 0:
        return np.empty((0,), dtype=x.dtype)
    try:
        return np.take(x, indices, axis=axis)
    except TypeError:
        return np.take(x, indices.astype(int), axis=axis)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
gather_op = GatherOp(mgr)

# Case 1: 一维 data，一维 indices，axis=0
print("Case 1")
numpy_in = np.random.random((1023,)).astype(np.float32)
indices = np.arange(0, 512, 2).astype(np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=0)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 0
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 2: 二维 data，二维 indices，axis=1
print("Case 2")
numpy_in = np.random.random((1023, 15)).astype(np.float32)
indices = np.random.randint(0, 15, size=(3, 5), dtype=np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 1
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 3: 三维 data，一维 indices，axis=-1
print("Case 3")
numpy_in = np.random.random((3, 1023, 1023)).astype(np.float32)
indices = np.random.randint(0, 1023, size=(512,), dtype=np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=-1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = -1
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 4: 四维 data，多维 indices，axis=2
print("Case 4")
numpy_in = np.random.random((3, 3, 255, 255)).astype(np.float32)
indices = np.random.randint(0, 255, size=(5, 5, 64), dtype=np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=2)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 2
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 5: indices 为标量
print("Case 5")
numpy_in = np.random.random((225, 255)).astype(np.float32)
indices = np.array(127, dtype=np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 1
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 6: indices 为一维，data 为三维
print("Case 6")
numpy_in = np.random.random((3, 3, 225)).astype(np.float32)
indices = np.random.randint(0, 225, size=(128,), dtype=np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=2)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 2
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 7: indices 为三维
print("Case 7")
numpy_in = np.random.random((3, 3, 255)).astype(np.float32)
indices = np.random.randint(0, 255, size=(3, 3, 15), dtype=np.int32)

start_time = time.time()
np_out = np_gather(numpy_in, indices, axis=-1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = -1
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

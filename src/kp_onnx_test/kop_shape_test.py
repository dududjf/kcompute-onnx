from kp import Manager
import numpy as np
import time
from kp_onnx.kop_shape import ShapeOp


def np_shape(x, start=None, end=None):
    def _interval(n: int, start: int | None, end: int | None):
        if start == 0:
            if end is None or np.isnan(end):
                return None
            if end < 0:
                return (0, n + end)
            return (0, end)
        if end is None or np.isnan(end):
            return (start, n)
        if end < 0:
            return (start, n + end)
        return (start, end)

    ab = _interval(len(x.shape), start=start, end=end)
    if ab is None:
        return np.array(x.shape, dtype=np.int64)
    return np.array(x.shape[ab[0]:ab[1]], dtype=np.int64)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
shape_op = ShapeOp(mgr)

# Case 1: 无 start/end，返回完整 shape
print('Case 1')
numpy_in = np.random.random((3, 1023, 1023, 15)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = None
shape_op.end = None
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 2: start=0, end=None
print('Case 2')
numpy_in = np.random.random((3, 3, 255, 255, 1)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=0, end=None)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = 0
shape_op.end = None
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 3: start=0, end=-1
print('Case 3')
numpy_in = np.random.random((225, 255, 15)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=0, end=-1)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = 0
shape_op.end = -1
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 4: start=1, end=None
print('Case 4')
numpy_in = np.random.random((3, 1, 255, 15)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=1, end=None)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = 1
shape_op.end = None
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 5: start=2, end=-1
print('Case 5')
numpy_in = np.random.random((3, 3, 255, 255, 1)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=2, end=-1)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = 2
shape_op.end = -1
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 6: start=1, end=3
print('Case 6')
numpy_in = np.random.random((3, 1023, 255, 15)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=1, end=3)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = 1
shape_op.end = 3
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 7: scalar (0-dim tensor)
print('Case 7')
numpy_in = np.array(42.0, dtype=np.float32)  # shape=()
start_time = time.time()
np_out = np_shape(numpy_in)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = None
shape_op.end = None
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 8: start < 0
print('Case 8')
numpy_in = np.random.random((3, 1023, 15)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=-2)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = -2
shape_op.end = None
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 9: end > len(shape)
print('Case 9')
numpy_in = np.random.random((3, 3, 255, 255, 1)).astype(np.float32)
start_time = time.time()
np_out = np_shape(numpy_in, start=1, end=10)
print("Numpy:", np_out, time.time() - start_time, "seconds")

shape_op.start = 1
shape_op.end = 10
start_time = time.time()
kp_out = shape_op.run(numpy_in)[0]
print(f"{shape_op}:", kp_out, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

from kp import Manager
import numpy as np
import time
from kp_onnx.kop_concat import ConcatOp


def np_concat(*inputs, axis=0):
    def _preprocess(a: np.ndarray, axis: int) -> np.ndarray:
        if len(a.shape) == 0:
            raise RuntimeError(f"Concat: one input has an empty shape: {a!r}.")
        if axis >= len(a.shape):
            new_shape = a.shape + (1,) * (axis + 1 - len(a.shape))
            return a.reshape(new_shape)
        return a

    targs = tuple(_preprocess(a, axis) for a in inputs)
    return np.concatenate(targs, axis)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
concat_op = ConcatOp(mgr)

# Case 1: 基本 2D 拼接
print("Case 1")
concat_op.axis = 1
numpy_in1 = np.random.random((1023, 15)).astype(np.float32)
numpy_in2 = np.random.random((1023, 20)).astype(np.float32)

start_time = time.time()
np_out = np_concat(numpy_in1, numpy_in2, axis=concat_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = concat_op.run(numpy_in1, numpy_in2)[0]
print(f"{concat_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 2: 三个输入，3D 拼接
print("Case 2")
concat_op.axis = 2
numpy_in1 = np.random.random((3, 1023, 1023)).astype(np.float32)
numpy_in2 = np.random.random((3, 1023, 512)).astype(np.float32)
numpy_in3 = np.random.random((3, 1023, 256)).astype(np.float32)

start_time = time.time()
np_out = np_concat(numpy_in1, numpy_in2, numpy_in3, axis=concat_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = concat_op.run(numpy_in1, numpy_in2, numpy_in3)[0]
print(f"{concat_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 3: 1D 向量拼接
print("Case 3")
concat_op.axis = 0
numpy_in1 = np.random.random((1023,)).astype(np.float32)
numpy_in2 = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
np_out = np_concat(numpy_in1, numpy_in2, axis=concat_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = concat_op.run(numpy_in1, numpy_in2)[0]
print(f"{concat_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 4: 扩展维度拼接 (axis > rank)
print("Case 4")
concat_op.axis = 1
numpy_in1 = np.random.random((225,)).astype(np.float32)
numpy_in2 = np.random.random((225,)).astype(np.float32)

start_time = time.time()
np_out = np_concat(numpy_in1, numpy_in2, axis=concat_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = concat_op.run(numpy_in1, numpy_in2)[0]
print(f"{concat_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 5: 高维拼接 (4D)
print("Case 5")
concat_op.axis = 3
numpy_in1 = np.random.random((3, 3, 225, 255)).astype(np.float32)
numpy_in2 = np.random.random((3, 3, 225, 128)).astype(np.float32)

start_time = time.time()
np_out = np_concat(numpy_in1, numpy_in2, axis=concat_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = concat_op.run(numpy_in1, numpy_in2)[0]
print(f"{concat_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 6: 多输入 (五个输入拼接)
print("Case 6")
concat_op.axis = 1
inputs = [np.random.random((3, 255, 225, 15)).astype(np.float32) for _ in range(5)]

start_time = time.time()
np_out = np_concat(*inputs, axis=concat_op.axis)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = concat_op.run(*inputs)[0]
print(f"{concat_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

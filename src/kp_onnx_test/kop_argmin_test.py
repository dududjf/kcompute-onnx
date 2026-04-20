from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_argmin import ArgMinOp, DEFAULT_AXIS


def onnx_argmin(data, axis=0, keepdims=True, select_last_index=0):

    def _argmin(data, axis=0, keepdims=True):
        result = np.argmin(data, axis=axis)
        if keepdims and len(result.shape) < len(data.shape):
            result = np.expand_dims(result, axis)
        return result.astype(np.int64)

    def _argmin_use_numpy_select_last_index(data, axis=0, keepdims=True):
        data = np.flip(data, axis)
        result = np.argmin(data, axis=axis)
        result = data.shape[axis] - result - 1
        if keepdims:
            result = np.expand_dims(result, axis)
        return result.astype(np.int64)

    if select_last_index == 0:  # type: ignore
        return _argmin(data, axis=axis, keepdims=keepdims)
    return _argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

argmin_op = ArgMinOp(mgr)

# Case 1: 1D 张量，默认
print("Case 1: 1D 张量，默认")
numpy_in = np.random.uniform(-8, 8, (1024,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = DEFAULT_AXIS
argmin_op.keepdims = True
argmin_op.select_last_index = False
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 3D 张量，默认
print("Case 2: 3D 张量，默认")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = DEFAULT_AXIS
argmin_op.keepdims = True
argmin_op.select_last_index = False
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 3D 张量，轴 1
print("Case 3: 3D 张量，轴 1")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=1)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = 1
argmin_op.keepdims = True
argmin_op.select_last_index = False
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 3D 张量，轴 2
print("Case 4: 3D 张量，轴 2")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=2)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = 2
argmin_op.keepdims = True
argmin_op.select_last_index = False
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: 3D 张量，轴 2，不保持维度
print("Case 5: 3D 张量，轴 2，不保持维度")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=2, keepdims=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = 2
argmin_op.keepdims = False
argmin_op.select_last_index = False
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: 3D 张量，负轴
print("Case 6: 3D 张量，负轴")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, axis=-2)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = -2
argmin_op.keepdims = True
argmin_op.select_last_index = False
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: 3D 张量，选择最后一次出现的索引
print("Case 7: 3D 张量，选择最后一次出现的索引")
numpy_in = np.random.uniform(-8, 8, (128, 256, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_argmin(numpy_in, select_last_index=1)[0]
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
argmin_op.axis = DEFAULT_AXIS
argmin_op.keepdims = True
argmin_op.select_last_index = True
kp_out = argmin_op.run(numpy_in)[0]
print(f"{argmin_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()
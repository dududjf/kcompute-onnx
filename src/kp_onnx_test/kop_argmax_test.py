from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_argmax import ArgMaxOp


def numpy_argmax(x: np.ndarray, axis: int = 0, keepdims=True, select_last_index=False) -> np.ndarray:
    if not select_last_index:
        result = np.argmax(x, axis=axis)
        if keepdims and len(result.shape) < len(x.shape):
            result = np.expand_dims(result, axis=axis)
        return result.astype(np.int64)
    else:
        data = np.flip(x, axis)
        result = np.argmax(data, axis=axis)
        result = data.shape[axis] - result - 1
        if keepdims:
            result = np.expand_dims(result, axis=axis)
        return result.astype(np.int64)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

argmax_op = ArgMaxOp(mgr)
x = np.random.uniform(-3.0, 3.0, (32, 64, 128)).astype(np.float32)

# -------- Case 1 --------
print("Case 1 for not axis and not keepdims and not select_last_index")
# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=0, keepdims=True, select_last_index=False)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
argmax_op.axis = 0
argmax_op.keepdims = True
argmax_op.select_last_index = False
kp_out = argmax_op.run(x)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

# Evaluation
print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2 --------
print("Case 2 for not axis and not keepdims and select_last_index")
# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=0, keepdims=True, select_last_index=True)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
argmax_op.axis = 0
argmax_op.keepdims = True
argmax_op.select_last_index = True
kp_out = argmax_op.run(x)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

# Evaluation
print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3 --------
print("Case 3 for not axis and keepdims is 0 and not select_last_index")
# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=0, keepdims=False, select_last_index=False)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
argmax_op.axis = 0
argmax_op.keepdims = False
argmax_op.select_last_index = False
kp_out = argmax_op.run(x)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4 --------
print("Case 4 for axis is 2 and keepdims and select_last_index")
x[:, :, 127] = 10  # 在 axis=2 的最后位置设最大值
x[:, :, 126] = 10  # 在 axis=2 的倒数第二位置设最大值

# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=2, keepdims=False, select_last_index=True)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
argmax_op.axis = 2
argmax_op.keepdims = False
argmax_op.select_last_index = True
kp_out = argmax_op.run(x)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

# Evaluation
print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5 --------
print("Case 5 for axis is -1 and keepdims and select_last_index")

# NumPy
start_time = time.time()
np_out = numpy_argmax(x, axis=-1, keepdims=False, select_last_index=True)
print(f"Numpy time: {time.time() - start_time} seconds")

# Kompute
start_time = time.time()
argmax_op.axis = -1
argmax_op.keepdims = False
argmax_op.select_last_index = True
kp_out = argmax_op.run(x)[0]
print(f"{argmax_op} time: {time.time() - start_time} seconds")

# Evaluation
print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
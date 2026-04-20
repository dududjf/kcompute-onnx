from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_eyelike import EyeLikeOp

DEFAULT_K = 0

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

eyelike_op = EyeLikeOp(mgr)


def onnx_eyelike(x: np.ndarray, k: int) -> np.ndarray:
    shape = x.shape
    if len(shape) == 1:
        n = shape[0]
        return np.eye(n, n, k=k, dtype=np.float32)
    elif len(shape) == 2:
        n, m = shape
        return np.eye(n, m, k=k, dtype=np.float32)
    else:
        raise ValueError(f"EyeLike only supports 1D or 2D input, got {shape}")


# Case 1: 1D 输入 默认k=0
print("\nCase 1: 1D input")
x = np.random.uniform(-5, 5, (1024,)).astype(np.float32)

t0 = time.time()
numpy_out = onnx_eyelike(x, DEFAULT_K)
print("NumPy:", numpy_out.shape, time.time() - t0, "seconds")

t1 = time.time()
kp_out = eyelike_op.run(x)[0]
print(f"{eyelike_op}:", kp_out.shape, time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 2: 2D 输入 默认k=0
print("\nCase 2: 2D input")
x = np.random.uniform(-5, 5, (1024, 2048)).astype(np.float32)

t0 = time.time()
numpy_out = onnx_eyelike(x, DEFAULT_K)
print("NumPy:", numpy_out.shape, time.time() - t0, "seconds")

t1 = time.time()
kp_out = eyelike_op.run(x)[0]
print(f"{eyelike_op}:", kp_out.shape, time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 3: 1D 输入 指定k=2
print("\nCase 3: 1D input k=2")
x = np.random.uniform(-5, 5, (1024,)).astype(np.float32)

k = 2
t0 = time.time()
numpy_out = onnx_eyelike(x, k)
print("NumPy:", numpy_out.shape, time.time() - t0, "seconds")

t1 = time.time()
eyelike_op.k = k
kp_out = eyelike_op.run(x)[0]
print(f"{eyelike_op}:", kp_out.shape, time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 4: 2D 输入 指定k=-1
print("\nCase 4: 2D input k=-1")
x = np.random.uniform(-5, 5, (1024, 2048)).astype(np.float32)

k = -1
t0 = time.time()
numpy_out = onnx_eyelike(x, k)
print("NumPy:", numpy_out.shape, time.time() - t0, "seconds")

t1 = time.time()
eyelike_op.k = k
kp_out = eyelike_op.run(x)[0]
print(f"{eyelike_op}:", kp_out.shape, time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
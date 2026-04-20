from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_trilu import TriluOp, DEFAULT_K

DEFAULT_UPPER = 1

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

trilu_op = TriluOp(mgr)


def trilu_numpy(x: np.ndarray, k: int, upper: int) -> np.ndarray:
    """
    - upper=1 → 上三角：np.triu(x, k)
    - upper=0 → 下三角：np.tril(x, k)
    """
    if upper:
        return np.triu(x, k).astype(x.dtype, copy=False)
    else:
        return np.tril(x, k).astype(x.dtype, copy=False)


# Case 1: 默认（默认k=0，UPPER=1上三角）
print("\nCase 1: default (upper=DEFAULT_UPPER, k=DEFAULT_K)")
x = np.random.uniform(-5, 5, (640, 10240)).astype(np.float32)  # 最后两维视作矩阵(640 x 10240)

t0 = time.time()
numpy_out = trilu_numpy(x, DEFAULT_K, DEFAULT_UPPER)
print("NumPy (Default):", time.time() - t0, "seconds")

t1 = time.time()
kp_out = trilu_op.run(x)[0]
print(f"{trilu_op} (Default):", time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 2: 指定 k=+3 正偏移：更靠上对角线
print("\nCase 2: k=+3, upper=DEFAULT_UPPER")
x = np.random.uniform(-3, 3, (5120, 5120)).astype(np.float32)

k = 3
t0 = time.time()
numpy_out = trilu_numpy(x, k, DEFAULT_UPPER)
print("NumPy (k=+3):", time.time() - t0, "seconds")

t1 = time.time()
kp_out = trilu_op.run(x, k)[0]
print(f"{trilu_op} (k=+3):", time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 3: 指定下三角（upper=0），k默认
print("\nCase 3: lower (upper=0), k=DEFAULT_K")
x = np.random.uniform(-4, 4, (2560, 2560)).astype(np.float32)

upper = 0  # 下三角
t0 = time.time()
numpy_out = trilu_numpy(x, DEFAULT_K, upper)
print("NumPy (lower):", time.time() - t0, "seconds")

t1 = time.time()
trilu_op.upper = upper
kp_out = trilu_op.run(x)[0]
print(f"{trilu_op} (lower):", time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 4: 下三角 + k=-3
print("\nCase 4: lower (upper=0), k=-3")
x = np.random.uniform(-2, 2, (3200, 2400)).astype(np.float32)  # 矩形矩阵

k = -3
upper = 0
t0 = time.time()
numpy_out = trilu_numpy(x, k, upper)
print("NumPy (lower, k=-2):", time.time() - t0, "seconds")

t1 = time.time()
trilu_op.upper = upper
kp_out = trilu_op.run(x, k)[0]
print(f"{trilu_op} (lower, k=-2):", time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 5: 批次（*,N,M）
print("\nCase 5: batch （*,N,M）")
x = np.random.uniform(-1, 1, (2, 3, 32, 48)).astype(np.float32)

t0 = time.time()
numpy_out = trilu_numpy(x, DEFAULT_K, DEFAULT_UPPER)
print("NumPy （*,N,M）:", time.time() - t0, "seconds")

t1 = time.time()
trilu_op.upper = DEFAULT_UPPER
kp_out = trilu_op.run(x)[0]
print(f"{trilu_op} （*,N,M）:", time.time() - t1, "seconds")

print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
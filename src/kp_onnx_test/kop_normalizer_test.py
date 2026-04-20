import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_normalizer import NormalizerOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

normalizer_op = NormalizerOp(mgr)


def np_normalizer(x, norm=None):
    if norm == "MAX":
        div = np.abs(x).max(axis=1).reshape((x.shape[0], -1))
        return x / np.maximum(div, 1e-30)
    elif norm == "L1":
        div = np.abs(x).sum(axis=1).reshape((x.shape[0], -1))
        return x / np.maximum(div, 1e-30)
    elif norm == "L2":
        xn = np.square(x).sum(axis=1)
        np.sqrt(xn, out=xn)
        norm = np.maximum(xn.reshape((x.shape[0], -1)), 1e-30)
        return x / norm
    else:
        raise ValueError(f"Unexpected value for norm='{norm}'.")


x = np.random.random((1024, 512)).astype(np.float32)

# -------- Case 1: Simple 2D matrix, MAX norm --------
print("Case 1: 1D matrix, MAX norm")
start_time = time.time()
np_out = np_normalizer(x, 'MAX')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'MAX'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: L1 norm --------
print("Case 2: 2D matrix, L1 norm")
start_time = time.time()
np_out = np_normalizer(x, 'L1')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'L1'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: L2 norm --------
print("Case 3: 2D matrix, L2 norm")
start_time = time.time()
np_out = np_normalizer(x, 'L2')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'L2'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: Edge case - zeros --------
print("Case 4: Edge case - row with all zeros")
x = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]], dtype=np.float32)

start_time = time.time()
np_out = np_normalizer(x, 'MAX')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'MAX'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: Edge case - zeros --------
print("Case 5: Edge case - row with all zeros")
x = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]], dtype=np.float32)

start_time = time.time()
np_out = np_normalizer(x, 'L1')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'L1'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6: Edge case - zeros --------
print("Case 6: Edge case - row with all zeros")
x = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]], dtype=np.float32)

start_time = time.time()
np_out = np_normalizer(x, 'L2')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'L2'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 7: 1D input branch --------
print("Case 7: 1D input, MAX norm")
x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

start_time = time.time()
np_out = np_normalizer(x.reshape(1,-1), 'MAX')
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
normalizer_op.norm = 'MAX'
kp_out = normalizer_op.run(x)[0]
print(f"{normalizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
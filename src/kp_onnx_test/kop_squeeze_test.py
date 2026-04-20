from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_squeeze import SqueezeOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

squeeze_op = SqueezeOp(mgr)


def onnx_squeeze(x: np.ndarray, axes=None) -> np.ndarray:
    if axes is None:
        return np.squeeze(x)
    else:
        return np.squeeze(x, axis=axes)


# Case 1: 默认 squeeze (所有维度==1的去掉)
print("\nCase 1: default squeeze")
x = np.random.uniform(-1, 1, (2, 1, 32, 1, 16, 1)).astype(np.float32)

t0 = time.time()
numpy_out = onnx_squeeze(x)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = squeeze_op.run(x)[0]
print(f"{squeeze_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))


# Case 2: 指定 axes = (1, 3)
print("\nCase 2: squeeze with axes=(1,3)")
x = np.random.uniform(-1, 1, (2, 1, 32, 1, 16, 1)).astype(np.float32)

axes = (1, 3)
t0 = time.time()
numpy_out = onnx_squeeze(x, axes)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = squeeze_op.run(x, np.array(axes, dtype=np.int32))[0]
print(f"{squeeze_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 3 : 指定负轴 axes = (-1, -3)
print("\nCase 3: squeeze with axes=(-1, -3)")
x = np.random.uniform(-1, 1, (2, 1, 32, 1, 16, 1)).astype(np.float32)

axes = (-1, -3)
t0 = time.time()
numpy_out = onnx_squeeze(x, axes)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = squeeze_op.run(x, np.array(axes, dtype=np.int32))[0]
print(f"{squeeze_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
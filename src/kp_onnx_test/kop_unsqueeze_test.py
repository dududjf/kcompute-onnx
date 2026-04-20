from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_unsqueeze import UnsqueezeOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

unsqueeze_op = UnsqueezeOp(mgr)


def onnx_unsqueeze(data: np.ndarray, axes=None) -> np.ndarray:
    if axes is not None:
        if hasattr(axes, "__iter__") and isinstance(axes, np.ndarray) and axes.ndim > 0:
            try:
                sq = np.expand_dims(data, axis=tuple(axes))
            except TypeError:
                if len(axes) == 1:
                    sq = np.expand_dims(data, axis=tuple(axes)[0])
                else:
                    sq = data
                    for a in reversed(axes):
                        sq = np.expand_dims(sq, axis=a)
        else:
            sq = np.expand_dims(data, axis=axes)
    else:
        raise RuntimeError(
            "axes cannot be None for operator Unsqueeze (Unsqueeze_13)."
        )
    return sq


# Case 1: 指定 axes=(1,3)
print("\nCase 1: 指定 axes=(1,3)")
x = np.random.uniform(-1, 1, (2, 32, 16)).astype(np.float32)

axes = (1, 3)

t0 = time.time()
numpy_out = onnx_unsqueeze(x, axes)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = unsqueeze_op.run(x, np.array(axes, dtype=np.int32))[0]
print(f"{unsqueeze_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

# Case 2: 指定负轴 axes=(-1,-3)
print("\nCase 2: 指定负轴 axes=(-1,-3)")
x = np.random.uniform(-1, 1, (2, 32, 16)).astype(np.float32)

axes = (-1, -3)

t0 = time.time()
numpy_out = onnx_unsqueeze(x, axes)
print("NumPy:", time.time() - t0, "seconds")

t1 = time.time()
kp_out = unsqueeze_op.run(x, np.array(axes, dtype=np.int32))[0]
print(f"{unsqueeze_op}:", time.time() - t1, "seconds")

print("numpy_in.shape:", x.shape)
print("numpy_out.shape:", numpy_out.shape)
print("kp_out.shape:", kp_out.shape)
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
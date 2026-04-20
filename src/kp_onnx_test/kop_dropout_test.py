import time
import numpy as np
from kp import Manager
from kp_onnx_ssbo.kop_dropout import DropoutOp


def dropout_numpy(x, ratio=0.5, training_mode=False, seed=None):
    if (not training_mode) or ratio == 0.0:
        return x
    rnd = np.random.RandomState(None if seed is None else int(seed))
    keep = rnd.uniform(0.0, 1.0, size=x.shape) >= ratio
    scale = 1.0 / (1.0 - ratio) if ratio < 1.0 else 0.0
    y = (keep.astype(np.float32) * x.astype(np.float32)) * scale
    return y


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

dropout_op = DropoutOp(mgr)

# ---------------- Case 1: 3D Tensor, ratio: None, traning_mode: None, seed: None ----------------
print("Case 1: 3D Tensor, ratio: None, traning_mode: None, seed: None")
x = np.random.random((32, 512, 1024)).astype(np.float32)

start_time = time.time()
np_out = dropout_numpy(x, seed=None)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
dropout_op.seed = None
kp_out = dropout_op.run(x)[0]
print(f"{dropout_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 2: 3D Tensor, ratio: 0.3, traning_mode: True, seed: None ----------------
print("Case 2: 3D Tensor, ratio: 0.3, traning_mode: True, seed: None")
x = np.random.random((32, 512, 2048)).astype(np.float32)
ratio = 0.3

start_time = time.time()
np_out = dropout_numpy(x, ratio=ratio, training_mode=True, seed=None)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
dropout_op.seed = None
kp_out = dropout_op.run(x, ratio, True)[0]
print(f"{dropout_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 3: 3D Tensor, ratio: 0.5, traning_mode: True, seed: 42 ----------------
print("Case 3: 3D Tensor, ratio: 0.5, traning_mode: True, seed: 42")
x = np.random.random((512, 512)).astype(np.float32)
ratio = 0.5
seed = 42

start_time = time.time()
np_out = dropout_numpy(x, ratio=ratio, training_mode=True, seed=seed)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
dropout_op.seed = seed
kp_out = dropout_op.run(x, ratio, True)
print(f"{dropout_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 4: 1D Tensor, ratio: 0, traning_mode: True, seed: None ----------------
print("Case 4: 1D Tensor, ratio: 0, traning_mode: True, seed: None")
x = np.random.random((2048,)).astype(np.float32)

start_time = time.time()
np_out = dropout_numpy(x, ratio=0.0, training_mode=True, seed=None)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
dropout_op.seed = None
kp_out = dropout_op.run(x, 0.0, True)[0]
print(f"{dropout_op} (ratio=0): ", time.time() - start_time, "seconds")
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

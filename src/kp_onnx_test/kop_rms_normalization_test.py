from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_rms_normalization import RmsNormalizationOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

rms_normalization_op = RmsNormalizationOp(mgr)


def np_rms_norm(X, Scale, axis=-1, epsilon=1e-05, stash_type=1):
    if stash_type != 1:
        raise NotImplementedError(
            f"RMSNormalization not implemented for stash_type={stash_type} != 1."
        )
    shape = X.shape
    rank = len(shape)
    if axis < 0:
        axis = axis + rank

    # This computes RMS for every x_mat's column.
    x_squared = np.power(X, 2)
    x_squared_mean = np.mean(x_squared, axis=tuple(range(axis, len(shape))), keepdims=True)
    # epsilon adjustment to avoid divide-by-zero.
    rmseps = x_squared_mean + epsilon
    # This computes RMS for every x_mat's column.
    rms = np.sqrt(rmseps)
    rms_reciprocal = np.reciprocal(rms)
    y_mat = X * rms_reciprocal
    Y = y_mat * Scale

    return Y


# -------- Case 1: data: 1D --------
print("Case 1: data: 1D")
x = np.random.random((32,)).astype(np.float32)
scale = np.random.random((1,)).astype(np.float32)

start_time = time.time()
np_out = np_rms_norm(x, scale)
print("NumPy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
rms_normalization_op.axis = -1
rms_normalization_op.epsilon = 1e-05
kp_out = rms_normalization_op.run(x, scale)[0]
print(f"{rms_normalization_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: data: 2D --------
print("Case 2: data: 2D")
x = np.random.random((32, 511)).astype(np.float32)
scale = np.random.random((511,)).astype(np.float32)

start_time = time.time()
np_out = np_rms_norm(x, scale)
print("NumPy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
rms_normalization_op.axis = -1
rms_normalization_op.epsilon = 1e-05
kp_out = rms_normalization_op.run(x, scale)[0]
print(f"{rms_normalization_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: data: 4D, axis: 0, epsilon: 1e-03 --------
print("Case 3: data: 4D, axis: 0, epsilon: 1e-03")
x = np.random.random((32, 32, 16, 8)).astype(np.float32)
scale = np.random.random((1, 32, 16, 1)).astype(np.float32)

start_time = time.time()
np_out = np_rms_norm(x, scale, axis=0, epsilon=1e-03, stash_type=1)
print("NumPy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
rms_normalization_op.axis = 0
rms_normalization_op.epsilon = 1e-03
kp_out = rms_normalization_op.run(x, scale)[0]
print(f"{rms_normalization_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
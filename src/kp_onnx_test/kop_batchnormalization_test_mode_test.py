from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_batchnormalization_test_mode import BatchNormalizationTestModeOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

batch_norm_test_op = BatchNormalizationTestModeOp(mgr)

def _batchnorm_test_mode(
    x: np.ndarray,
    s: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    y = s * (x - mean) / np.sqrt(var + epsilon) + bias
    return y.astype(x.dtype)  # type: ignore


def _batchnorm_training_mode(
    x: np.ndarray,
    s: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
) -> np.ndarray:
    axis = tuple(np.delete(np.arange(len(x.shape)), 1))
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(x, s, bias, saved_mean, saved_var, epsilon=epsilon)
    return (
        y.astype(x.dtype),
        saved_mean.astype(x.dtype),
        saved_var.astype(x.dtype),
        output_mean.astype(x.dtype),
        output_var.astype(x.dtype),
    )


def np_batch_norm_test(x, scale, bias, mean, var, epsilon=None, momentum=None, training_mode=None):
    if training_mode == 0:
        res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
        return res
    res, __, _, output_mean, output_var = _batchnorm_training_mode(
        x, scale, bias, mean, var, momentum, epsilon
    )
    return res, output_mean, output_var


# -------- Case 1: data: 4D --------
print("Case 1: data: 4D")
x = np.random.random((32, 8, 64, 64)).astype(np.float32)
scale = np.random.random((8,)).astype(np.float32)
bias = np.random.random((8,)).astype(np.float32)
mean = np.random.random((8,)).astype(np.float32)
var = np.random.random((8,)).astype(np.float32)

start_time = time.time()
np_out = np_batch_norm_test(x, scale, bias, mean, var, epsilon=1e-05, momentum=0.9, training_mode=0)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
batch_norm_test_op.epsilon = 1e-05
batch_norm_test_op.momentum = 0.9
batch_norm_test_op.training_mode = 0
kp_out = batch_norm_test_op.run(x, scale, bias, mean, var)[0]
print(f"{batch_norm_test_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: data: 6D --------
print("Case 2: data: 6D")
x = np.random.random((16, 8, 8, 32, 32, 32)).astype(np.float32)
scale = np.random.random((8,)).astype(np.float32)
bias = np.random.random((8,)).astype(np.float32)
mean = np.random.random((8,)).astype(np.float32)
var = np.random.random((8,)).astype(np.float32)

start_time = time.time()
np_out = np_batch_norm_test(x, scale, bias, mean, var, epsilon=1e-06, momentum=0.5, training_mode=0)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
batch_norm_test_op.epsilon = 1e-06
batch_norm_test_op.momentum = 0.5
batch_norm_test_op.training_mode = 0
kp_out = batch_norm_test_op.run(x, scale, bias, mean, var)[0]
print(f"{batch_norm_test_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: data: 2D --------
print("Case 3: data: 2D")
x = np.random.random((64, 8)).astype(np.float32)
scale = np.random.random((8,)).astype(np.float32)
bias = np.random.random((8,)).astype(np.float32)
mean = np.random.random((8,)).astype(np.float32)
var = np.random.random((8,)).astype(np.float32)

start_time = time.time()
np_out = np_batch_norm_test(x, scale, bias, mean, var, epsilon=1e-05, momentum=0.9, training_mode=0)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
batch_norm_test_op.epsilon = 1e-05
batch_norm_test_op.momentum = 0.9
batch_norm_test_op.training_mode = 0
kp_out = batch_norm_test_op.run(x, scale, bias, mean, var)[0]
print(f"{batch_norm_test_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
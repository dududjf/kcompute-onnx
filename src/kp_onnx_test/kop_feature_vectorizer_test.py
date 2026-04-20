import time
import numpy as np
from kp import Manager
from kp_onnx_ssbo.kop_feature_vectorizer import FeatureVectorizerOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

feature_vectorizer_op = FeatureVectorizerOp(mgr)


def _preprocess(a, cut):
    if len(a.shape) == 1:
        a = a.reshape((-1, 1))
    if len(a.shape) != 2:
        raise ValueError(f"Every input must have 1 or 2 dimensions not {a.shape}.")
    if cut < a.shape[1]:
        return a[:, :cut]
    if cut > a.shape[1]:
        b = np.zeros((a.shape[0], cut), dtype=a.dtype)
        b[:, : a.shape[1]] = a
        return b
    return a


def _run(*args, inputdimensions=None):
    args = [
        _preprocess(a, axis) for a, axis in zip(args, inputdimensions)
    ]
    res = np.concatenate(args, axis=1)
    return res


print("Case 1:")
# Case 1: two 2D inputs, truncation and padding
data1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
data2 = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)
inputdimensions = [0, 0]  # truncate data1 to 2, pad data2 to 5

start_time = time.time()
np_out = _run(data1, data2, inputdimensions=inputdimensions)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
feature_vectorizer_op.inputdimensions = inputdimensions
kp_out = feature_vectorizer_op.run(data1, data2)[0]
print("kp_out:", kp_out)
print(f"{feature_vectorizer_op}: ", time.time() - start_time, "seconds")

# print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 2: one 1D input and one 2D input
print("Case 2:")
data1 = np.random.random((5,)).astype(np.float32)
data2 = np.random.random((5, 2)).astype(np.float32)
inputdimensions = [-1, 3]

start_time = time.time()
np_out = _run(data1, data2, inputdimensions=inputdimensions)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
feature_vectorizer_op.inputdimensions = inputdimensions
kp_out = feature_vectorizer_op.run(data1, data2)[0]
print(f"{feature_vectorizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 3: three inputs, varying shapes
print("Case 3:")
data1 = np.random.random((3, 1)).astype(np.float32)
data2 = np.random.random((3, 4)).astype(np.float32)
data3 = np.random.random((3,)).astype(np.float32)
inputdimensions = [20, -1, 5]

start_time = time.time()
np_out = _run(data1, data2, data3, inputdimensions=inputdimensions)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
feature_vectorizer_op.inputdimensions = inputdimensions
kp_out = feature_vectorizer_op.run(data1, data2, data3)[0]
print(f"{feature_vectorizer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

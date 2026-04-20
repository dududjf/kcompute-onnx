import time
import numpy as np
from kp import Manager
from kp_onnx_ssbo.kop_array_feature_extractor import ArrayFeatureExtractorOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

array_feature_extractor_op = ArrayFeatureExtractorOp(mgr)


def _array_feature_extrator(data, indices):
    if len(indices.shape) == 2 and indices.shape[0] == 1:
        index = indices.ravel().tolist()
        add = len(index)
    elif len(indices.shape) == 1:
        index = indices.tolist()
        add = len(index)
    else:
        add = 1
        for s in indices.shape:
            add *= s
        index = indices.ravel().tolist()
    if len(data.shape) == 1:
        new_shape = (1, add)
    else:
        new_shape = [*data.shape[:-1], add]
    try:
        tem = data[..., index]
    except IndexError as e:
        raise RuntimeError(f"data.shape={data.shape}, indices={indices}") from e
    res = tem.reshape(new_shape)
    return res

# Case 1: 2D data with 1D indices
print("Case 1: 2D data + 1D indices")
data = np.random.random((4, 6)).astype(np.float32)
indices = np.array([0, 2, -5], dtype=np.int64)

start_time = time.time()
np_out = _array_feature_extrator(data, indices)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = array_feature_extractor_op.run(data, indices)[0]
print(f"{array_feature_extractor_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 2: 1D data with scalar indices reshape (1, n)
print("Case 2: 1D data + 2D indices")
data = np.random.random((8,)).astype(np.float32)
indices = np.array([[1, 3, 4]], dtype=np.int64)

start_time = time.time()
np_out = _array_feature_extrator(data, indices)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = array_feature_extractor_op.run(data, indices)[0]
print(f"{array_feature_extractor_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# Case 3: 3D data with multi-dimensional indices
print("Case 3: 3D data + multi-d indices")
data = np.random.random((2, 3, 5)).astype(np.float32)
indices = np.array([[[0, -1], [-5, 0]], [[1, -3], [-5, 0]]], dtype=np.int64)

start_time = time.time()
np_out = _array_feature_extrator(data, indices)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = array_feature_extractor_op.run(data, indices)[0]
print(f"{array_feature_extractor_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

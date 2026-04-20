from kp import Manager
import numpy as np
import time
from kp_onnx.kop_topk import TopKOp


def topk_sorted_implementation(X, k, axis, largest):
    if isinstance(k, np.ndarray):
        if k.size != 1:
            raise RuntimeError(f"k must be an integer not {k!r}.")
        k = k[0]
    k = int(k)

    ind_axis = np.indices(X.shape)[axis]
    if largest:
        ind_axis = -ind_axis

    sorted_indices = np.lexsort((ind_axis, X), axis=axis)
    sorted_values = np.take_along_axis(X, sorted_indices, axis=axis)

    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)

    ark = np.arange(k)
    topk_sorted_indices = np.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = np.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


def np_topk(data, k_tensor, axis, largest=1):
    k_val = k_tensor.flatten()[0]
    ndim = data.ndim
    if axis < 0:
        axis += ndim

    sort_val, sort_ind = topk_sorted_implementation(data, k_val, axis, largest)
    return sort_val, sort_ind.astype(np.int64)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
topk_op = TopKOp(mgr, axis=-1, largest=1, sorted=1)

# Case 1: Standard Float32, Axis = -1, Largest = 1
print('\n=== Case 1: Standard Float32 ===')
topk_op.axis = -1
topk_op.largest = 1

numpy_in = np.random.random((3, 128, 256)).astype(np.float32)
k_in = np.array([10], dtype=np.int64)

start_time = time.time()
np_out_val, np_out_ind = np_topk(numpy_in, k_in, topk_op.axis, topk_op.largest)
print("Numpy:", np_out_val.shape, f"{time.time() - start_time:.6f}s")

start_time = time.time()
kp_out = topk_op.run(numpy_in, k_in)
kp_out_val, kp_out_ind = kp_out[0], kp_out[1]
print(f"Kompute:", kp_out_val.shape, f"{time.time() - start_time:.6f}s")

print("Values correct:", np.allclose(np_out_val, kp_out_val, rtol=1e-5, atol=1e-5))
print("Indices correct:", np.array_equal(np_out_ind, kp_out_ind))


# Case 2: Int32 Input, Axis = 1, Largest = 1
print('\n=== Case 2: Int32 Input, Axis=1 ===')
topk_op.axis = 1
topk_op.largest = 1

numpy_in = np.random.randint(-10000, 10000, (16, 512, 32)).astype(np.int32)
k_in = np.array([5], dtype=np.int64)

start_time = time.time()
np_out_val, np_out_ind = np_topk(numpy_in, k_in, topk_op.axis, topk_op.largest)
print("Numpy:", np_out_val.shape, f"{time.time() - start_time:.6f}s")

start_time = time.time()
kp_out = topk_op.run(numpy_in, k_in)
kp_out_val, kp_out_ind = kp_out[0], kp_out[1]
print(f"Kompute:", kp_out_val.shape, f"{time.time() - start_time:.6f}s")

print("Values correct:", np.array_equal(np_out_val, kp_out_val))
print("Indices correct:", np.array_equal(np_out_ind, kp_out_ind))


# Case 3: Smallest Mode (Largest = 0), Axis = 0
print('\n=== Case 3: Smallest Mode (Largest=0), Axis=0 ===')
topk_op.axis = 0
topk_op.largest = 0

numpy_in = np.random.random((256, 64, 64)).astype(np.float32)
k_in = np.array([20], dtype=np.int64)

start_time = time.time()
np_out_val, np_out_ind = np_topk(numpy_in, k_in, topk_op.axis, topk_op.largest)
print("Numpy:", np_out_val.shape, f"{time.time() - start_time:.6f}s")

start_time = time.time()
kp_out = topk_op.run(numpy_in, k_in)
kp_out_val, kp_out_ind = kp_out[0], kp_out[1]
print(f"Kompute:", kp_out_val.shape, f"{time.time() - start_time:.6f}s")

print("Values correct:", np.allclose(np_out_val, kp_out_val, rtol=1e-5, atol=1e-5))
print("Indices correct:", np.array_equal(np_out_ind, kp_out_ind))


# Case 4: Tie-Breaking Check (Explicit Repeats)
print('\n=== Case 4: Tie-Breaking (Stability) ===')
topk_op.axis = -1
topk_op.largest = 1

raw_data = [10, 20, 10, 30, 10]
numpy_in = np.array([raw_data] * 100).astype(np.float32)
k_in = np.array([5], dtype=np.int64)

start_time = time.time()
np_out_val, np_out_ind = np_topk(numpy_in, k_in, topk_op.axis, topk_op.largest)
print("Numpy:", np_out_val.shape, f"{time.time() - start_time:.6f}s")

start_time = time.time()
kp_out = topk_op.run(numpy_in, k_in)
kp_out_val, kp_out_ind = kp_out[0], kp_out[1]
print(f"Kompute:", kp_out_val.shape, f"{time.time() - start_time:.6f}s")

print("Values correct:", np.allclose(np_out_val, kp_out_val, rtol=1e-5, atol=1e-5))
print("Indices correct:", np.array_equal(np_out_ind, kp_out_ind))


# Case 5: Large K (Near dimension size)
print('\n=== Case 5: Large K (Near dim size) ===')
topk_op.axis = -1
topk_op.largest = 1

dim_size = 1024
numpy_in = np.random.random((100, dim_size)).astype(np.float32)
k_in = np.array([1000], dtype=np.int64)

start_time = time.time()
np_out_val, np_out_ind = np_topk(numpy_in, k_in, topk_op.axis, topk_op.largest)
print("Numpy:", np_out_val.shape, f"{time.time() - start_time:.6f}s")

start_time = time.time()
kp_out = topk_op.run(numpy_in, k_in)
kp_out_val, kp_out_ind = kp_out[0], kp_out[1]
print(f"Kompute:", kp_out_val.shape, f"{time.time() - start_time:.6f}s")

print("Values correct:", np.allclose(np_out_val, kp_out_val, rtol=1e-5, atol=1e-5))
print("Indices correct:", np.array_equal(np_out_ind, kp_out_ind))


# Case 6: 1D Array Input
print('\n=== Case 6: 1D Array Input ===')
topk_op.axis = 0
topk_op.largest = 1

numpy_in = np.random.random((256,)).astype(np.float32)
k_in = np.array([10], dtype=np.int64)

start_time = time.time()
np_out_val, np_out_ind = np_topk(numpy_in, k_in, topk_op.axis, topk_op.largest)
print("Numpy:", np_out_val.shape, f"{time.time() - start_time:.6f}s")

start_time = time.time()
kp_out = topk_op.run(numpy_in, k_in)
kp_out_val, kp_out_ind = kp_out[0], kp_out[1]
print(f"Kompute:", kp_out_val.shape, f"{time.time() - start_time:.6f}s")

print("Values correct:", np.allclose(np_out_val, kp_out_val, rtol=1e-5, atol=1e-5))
print("Indices correct:", np.array_equal(np_out_ind, kp_out_ind))

import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_unique import UniqueOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

unique_op = UniqueOp(mgr)


def _specify_int64(indices, inverse_indices, counts):  # type: ignore
    return (
        np.array(indices, dtype=np.int64),
        np.array(inverse_indices, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


def np_unique(x, axis=None, sorted=None):
    if axis is None or np.isnan(axis):
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)
    else:
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=axis)

    if not sorted:
        argsorted_indices = np.argsort(indices)
        inverse_indices_map = dict(
            zip(argsorted_indices, np.arange(len(argsorted_indices)))
        )
        indices = indices[argsorted_indices]
        y = np.take(x, indices, axis=0)
        inverse_indices = np.asarray(
            [inverse_indices_map[i] for i in inverse_indices], dtype=np.int64
        )
        counts = counts[argsorted_indices]

    indices, inverse_indices, counts = _specify_int64(
        indices, inverse_indices, counts
    )
    # numpy 2.0 has a different behavior than numpy 1.x.
    inverse_indices = inverse_indices.reshape(-1)
    return y, indices, inverse_indices, counts


print("Case 1:")
x = np.array([[[3, 1],
               [1, 2],
               [5, 6]],

              [[3, 1],
               [1, 2],
               [5, 6]]]).astype(np.float32)

start_time = time.time()
np_out = np_unique(x, axis=1, sorted=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
unique_op.axis = 1
unique_op.sorted = 1
kp_out = unique_op.run(x)
print(f"{unique_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("np_out:", np_out[i], "kp_out:", kp_out[i])
print('----')

print("Case 2:")
x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]]).astype(np.float32)

start_time = time.time()
np_out = np_unique(x, axis=0, sorted=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
unique_op.axis = 0
unique_op.sorted = 1
kp_out = unique_op.run(x)
print(f"{unique_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("np_out:", np_out[i], "kp_out:", kp_out[i])
print('----')

print("Case 3:")
x = np.array([[[1, 0, 0], [2, 0, 0], [2, 3, 4]], [[1, 0, 0], [1, 0, 0], [2, 3, 4]]]).astype(np.float32)

start_time = time.time()
np_out = np_unique(x, axis=1, sorted=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
unique_op.axis = 1
unique_op.sorted = 1
kp_out = unique_op.run(x)
print(f"{unique_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("np_out:", np_out[i], "kp_out:", kp_out[i])
print('----')

print("Case 4:")
x = np.array([[[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]],

              [[10, 11, 12],
               [13, 14, 15],
               [16, 17, 18]],

              [[19, 20, 21],
               [22, 23, 24],
               [25, 26, 27]]]).astype(np.float32)

start_time = time.time()
np_out = np_unique(x, axis=-2, sorted=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
unique_op.axis = -2
unique_op.sorted = 1
kp_out = unique_op.run(x)
print(f"{unique_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("np_out:", np_out[i], "kp_out:", kp_out[i])
print('----')

print("Case 5:")
x = np.random.random((8, 8, 8, 16)).astype(np.float32)

start_time = time.time()
np_out = np_unique(x, axis=None, sorted=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
unique_op.axis = None
unique_op.sorted = 1
kp_out = unique_op.run(x)
print(f"{unique_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("np_out:", np_out[i], "kp_out:", kp_out[i])
print('----')
import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_split import SplitOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

split_op = SplitOp(mgr)


def np_split(mat, split=None, axis=None, num_outputs=None):
    n_outputs = num_outputs or 1
    if split is None:
        if mat.shape[axis] % n_outputs == 0:
            div = mat.shape[axis] // n_outputs
            split = [div] * n_outputs
        else:
            div = mat.shape[axis] // n_outputs + 1
            split = [div] * n_outputs
            split[-1] += mat.shape[axis] - sum(split)

    sli = [slice(0, s) for s in mat.shape]
    res = []
    pos = 0
    for spl in split:
        sli[axis] = slice(pos, pos + spl)
        pos += spl
        res.append(mat[tuple(sli)])
    return res


x = np.random.random((32, 32, 8, 32)).astype(np.float32)

print("Case 1: not split, num_outputs is 8")
start_time = time.time()
np_out = np_split(x, axis=0, num_outputs=8)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_op.axis = 0
split_op.num_outputs = 8
kp_out = split_op.run(x)
print(f"{split_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print("Max error:", np.abs(np_out[i] - kp_out[i]).max())
    print(np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 2: not split, num_outputs is 3")
start_time = time.time()
np_out = np_split(x, axis=0, num_outputs=3)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_op.axis = 0
split_op.num_outputs = 3
kp_out = split_op.run(x)
print(f"{split_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print("Max error:", np.abs(np_out[i] - kp_out[i]).max())
    print(np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 3: not split, axis is -1, num_outputs is 8")
start_time = time.time()
np_out = np_split(x, axis=-1, num_outputs=8)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_op.axis = -1
split_op.num_outputs = 8
kp_out = split_op.run(x)
print(f"{split_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print("Max error:", np.abs(np_out[i] - kp_out[i]).max())
    print(np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 4: split with zero, axis is 2")
split = np.array([2, 0, 4, 2], dtype=np.int64)
axis = 2

start_time = time.time()
np_out = np_split(x, split=split, axis=axis, num_outputs=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_op.axis = axis
split_op.num_outputs = 1
kp_out = split_op.run(x, split)
print(f"{split_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print("Max error:", np.abs(np_out[i] - kp_out[i]).max())
    print(np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 5: all splits are zero")
split = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)  # 全0
axis = 2

start_time = time.time()
np_out = np_split(x, split=split, axis=axis, num_outputs=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_op.axis = axis
split_op.num_outputs = 1
kp_out = split_op.run(x, split)
print(f"{split_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print("shape:", kp_out[i].shape)
    print(np.array_equal(np_out[i], kp_out[i]))
print('----')

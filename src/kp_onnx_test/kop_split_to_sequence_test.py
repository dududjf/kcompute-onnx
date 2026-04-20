from kp import Manager
import time
import numpy as np
from kp_onnx_ssbo.kop_split_to_sequence import SplitToSequenceOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

split_to_seq_op = SplitToSequenceOp(mgr)


def np_split_to_sequence(mat, split=None, axis=0, keepdims=1):
    if split is None:
        split_length = [1 for _ in range(mat.shape[axis])]
    elif len(split.shape) == 0:
        # A scalar
        dim = mat.shape[axis]
        length = int(split)
        n = dim // int(length)
        split_length = [length] * n
        left = dim - length * n
        if left > 0:
            split_length.append(left)
    else:
        split_length = list(split)

    sli = [slice(0, s) for s in mat.shape]
    res = []
    pos = 0
    for spl in split_length:
        sli[axis] = slice(pos, pos + spl)
        pos += spl
        res.append(mat[tuple(sli)])

    if split is None and not keepdims:
        for i, res_i in enumerate(res):
            shape = list(res_i.shape)
            del shape[axis]
            res[i] = res_i.reshape(tuple(shape))
    return res


x = np.random.random((32, 32, 8, 32)).astype(np.float32)

print("Case 1: not split")
start_time = time.time()
np_out = np_split_to_sequence(x, axis=0, keepdims=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_to_seq_op.axis = 0
split_to_seq_op.keepdims = 1
kp_out = split_to_seq_op.run(x)
print(f"{split_to_seq_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print(f"Max error: ", np.abs(np_out[i] - kp_out[i]).max())
    print(f"All close: ", np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 2: split,axis=2")
split = np.array([2, 0, 0, 6], dtype=np.int64)
start_time = time.time()
np_out = np_split_to_sequence(x, split=split, axis=2, keepdims=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_to_seq_op.axis = 2
split_to_seq_op.keepdims = 1
kp_out = split_to_seq_op.run(x, split)
print(f"{split_to_seq_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print(f"Max error: ", np.abs(np_out[i] - kp_out[i]).max())
    print(f"All close: ", np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 3: split is scalar (not divisible), axis=-1，keepdims=0")
split = np.array(7, dtype=np.int64)
start_time = time.time()
np_out = np_split_to_sequence(x, split=split, axis=-1, keepdims=0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_to_seq_op.axis = -1
split_to_seq_op.keepdims = 0
kp_out = split_to_seq_op.run(x, split)
print(f"{split_to_seq_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    if 0 not in np_out[i].shape:
        print(f"Max error {i}: ", np.abs(np_out[i] - kp_out[i]).max())
    print(f"All close {i}: ", np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 4: split is None, keepdims=0")
start_time = time.time()
np_out = np_split_to_sequence(x, split=None, axis=1, keepdims=0)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_to_seq_op.axis = 1
split_to_seq_op.keepdims = 0
kp_out = split_to_seq_op.run(x)
print(f"{split_to_seq_op}: ", time.time() - start_time, "seconds")

for i in range(min(3, len(np_out))):
    if 0 not in np_out[i].shape:
        print(f"Max error {i}: ", np.abs(np_out[i] - kp_out[i]).max())
    print(f"All close {i}: ", np.allclose(np_out[i], kp_out[i], rtol=1e-4, atol=1e-4))
print('----')

print("Case 5: all splits are zero")
split = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
start_time = time.time()
np_out = np_split_to_sequence(x, split=split, axis=2, keepdims=1)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
split_to_seq_op.axis = 2
split_to_seq_op.keepdims = 1
kp_out = split_to_seq_op.run(x, split)
print(f"{split_to_seq_op}: ", time.time() - start_time, "seconds")

for i in range(len(np_out)):
    print(f"shape {i}: ", kp_out[i].shape)
    print(f"Equal {i}: ", np.array_equal(np_out[i], kp_out[i]))
print('----')
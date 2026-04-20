import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_one_hot_encoder import OneHotEncoderOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

one_hot_encoder_op = OneHotEncoderOp(mgr)


def np_one_hot_encoder(x, cats_int64s=None, cats_strings=None, zeros=None):
    if cats_int64s is not None and len(cats_int64s) > 0:
        classes = {v: i for i, v in enumerate(cats_int64s)}
    elif len(cats_strings) > 0:
        classes = {v: i for i, v in enumerate(cats_strings)}
    else:
        raise RuntimeError("No encoding was defined.")
    print("classes:", classes)

    shape = x.shape
    new_shape = (*shape, len(classes))
    res = np.zeros(new_shape, dtype=np.float32)
    if len(x.shape) == 1:
        for i, v in enumerate(x):
            j = classes.get(v, -1)
            if j >= 0:
                res[i, j] = 1.0
    elif len(x.shape) == 2:
        for a, row in enumerate(x):
            for i, v in enumerate(row):
                j = classes.get(v, -1)
                if j >= 0:
                    res[a, i, j] = 1.0
    else:
        raise RuntimeError(f"This operator is not implemented shape {x.shape}.")

    if not zeros:
        red = res.sum(axis=len(res.shape) - 1)
        if np.min(red) == 0:
            rows = []
            for i, val in enumerate(red):
                if val == 0:
                    rows.append({"row": i, "value": x[i]})
                    if len(rows) > 5:
                        break
            msg = "\n".join(str(_) for _ in rows)
            raise RuntimeError(
                f"One observation did not have any defined category.\n"
                f"classes: {classes}\nfirst rows:\n"
                f"{msg}\nres:\n{res[:5]}\nx:\n{x[:5]}"
            )

    return res


print("Case 1: 1D input")
x = np.array([1, 3, 5, 0, 2]).astype(np.float32)
cats_int64s = [0, 1, 2, 4, 5]
zeros = 1

start = time.time()
np_out = np_one_hot_encoder(x, cats_int64s=cats_int64s, zeros=zeros)
print("Numpy:", time.time() - start, "seconds")

start = time.time()
one_hot_encoder_op.cats_int64s = cats_int64s
one_hot_encoder_op.zeros = zeros
kp_out = one_hot_encoder_op.run(x)[0]
print(f"{one_hot_encoder_op}: ", time.time() - start, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 2: 1D input, zeros=0")
x = np.array([4, 10, 5, 1, 2]).astype(np.float32)
cats_int64s = [1, 10, 2, 4, 5]
zeros = 0

start = time.time()
np_out = np_one_hot_encoder(x, cats_int64s=cats_int64s, zeros=zeros)
print("Numpy:", time.time() - start, "seconds")

start = time.time()
one_hot_encoder_op.cats_int64s = cats_int64s
one_hot_encoder_op.zeros = zeros
kp_out = one_hot_encoder_op.run(x)[0]
print(f"{one_hot_encoder_op}: ", time.time() - start, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 3: 2D input")
x = np.array([[4, 10, 15, 1, 2], [4, 10, 5, 10, 2]]).astype(np.float32)
cats_int64s = [4, 10, 0, 1, 2, 5, 15]
zeros = 1

start = time.time()
np_out = np_one_hot_encoder(x, cats_int64s=cats_int64s, zeros=zeros)
print("Numpy:", time.time() - start, "seconds")

start = time.time()
one_hot_encoder_op.cats_int64s = cats_int64s
one_hot_encoder_op.zeros = zeros
kp_out = one_hot_encoder_op.run(x)[0]
print(f"{one_hot_encoder_op}: ", time.time() - start, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

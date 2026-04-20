from typing import Any

from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_concat_from_sequence import ConcatFromSequenceOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

concat_from_sequence_op = ConcatFromSequenceOp(mgr)


def np_concat_from_sequence(seq: list[Any], axis=None, new_axis=0):
    if seq is None:
        raise RuntimeError("A sequence cannot be null.")
    if new_axis == 1:
        if axis == -1:
            seq2 = [s[..., np.newaxis] for s in seq]
            res = np.concatenate(seq2, axis=-1)
        else:
            seq2 = [np.expand_dims(s, axis) for s in seq]
            res = np.concatenate(seq2, axis=axis)
    else:
        res = np.concatenate(seq, axis=axis)
    return res


# Case 1: 基本 2D 拼接
print("Case 1")
numpy_in1 = np.random.random((1023, 15)).astype(np.float32)
numpy_in2 = np.random.random((1023, 20)).astype(np.float32)

start_time = time.time()
np_out = np_concat_from_sequence([numpy_in1, numpy_in2], axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
concat_from_sequence_op.axis = 1
concat_from_sequence_op.new_axis = 0
kp_out = concat_from_sequence_op.run([numpy_in1, numpy_in2])[0]
print(f"{concat_from_sequence_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 2: 三个输入，3D 拼接
print("Case 2")
numpy_in1 = np.random.random((3, 1023, 1023)).astype(np.float32)
numpy_in2 = np.random.random((3, 1023, 512)).astype(np.float32)
numpy_in3 = np.random.random((3, 1023, 256)).astype(np.float32)

start_time = time.time()
np_out = np_concat_from_sequence([numpy_in1, numpy_in2, numpy_in3], axis=-1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
concat_from_sequence_op.axis = -1
concat_from_sequence_op.new_axis = 0
kp_out = concat_from_sequence_op.run([numpy_in1, numpy_in2, numpy_in3])[0]
print(f"{concat_from_sequence_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 3: 1D 向量拼接
print("Case 3")
numpy_in1 = np.random.random((1023,)).astype(np.float32)
numpy_in2 = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
np_out = np_concat_from_sequence([numpy_in1, numpy_in2], axis=-2, new_axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
concat_from_sequence_op.axis = -2
concat_from_sequence_op.new_axis = 1
kp_out = concat_from_sequence_op.run([numpy_in1, numpy_in2])[0]
print(f"{concat_from_sequence_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 4: 扩展维度拼接 (axis > rank)
print("Case 4")
numpy_in1 = np.random.random((225,)).astype(np.float32)
numpy_in2 = np.random.random((225,)).astype(np.float32)

start_time = time.time()
np_out = np_concat_from_sequence([numpy_in1, numpy_in2], axis=1, new_axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
concat_from_sequence_op.axis = 1
concat_from_sequence_op.new_axis = 1
kp_out = concat_from_sequence_op.run([numpy_in1, numpy_in2])[0]
print(f"{concat_from_sequence_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 5: 高维拼接 (4D)
print("Case 5")
numpy_in1 = np.random.random((3, 3, 225, 255)).astype(np.float32)
numpy_in2 = np.random.random((3, 3, 225, 255)).astype(np.float32)

start_time = time.time()
np_out = np_concat_from_sequence([numpy_in1, numpy_in2], axis=-5, new_axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
concat_from_sequence_op.axis = -5
concat_from_sequence_op.new_axis = 1
kp_out = concat_from_sequence_op.run([numpy_in1, numpy_in2])[0]
print(f"{concat_from_sequence_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 6: 多输入 (五个输入拼接)
print("Case 6")
inputs = [np.random.random((3, 255, 225, 15)).astype(np.float32) for _ in range(5)]

start_time = time.time()
np_out = np_concat_from_sequence(inputs, axis=-1, new_axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
concat_from_sequence_op.axis = -1
concat_from_sequence_op.new_axis = 1
kp_out = concat_from_sequence_op.run(inputs)[0]
print(f"{concat_from_sequence_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
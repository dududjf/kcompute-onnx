import time
import numpy as np
from kp import Manager
from kp_onnx_ssbo.kop_sequence_at import SequenceAtOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sequence_at_op = SequenceAtOp(mgr)


def onnx_sequence_at(seq, index):
    return seq[index]


print("Case 1: 正向索引，浮点数组")
seq = [np.random.random((4, 3)).astype(np.float32) for _ in range(5)]
pos = 2

strat_time = time.time()
np_out = onnx_sequence_at(seq, pos)
print("Numpy: ", time.time() - strat_time, "seconds")

strat_time = time.time()
kp_out = sequence_at_op.run(seq, pos)[0]
print(f"{sequence_at_op}: ", time.time() - strat_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 2: 负索引，浮点数组")
seq = [np.random.random((2, 5, 3)).astype(np.float32) for _ in range(4)]
pos = -1

np_out = onnx_sequence_at(seq, pos)
kp_out = sequence_at_op.run(seq, np.array(pos, dtype=np.int64))[0]
print(f"{sequence_at_op}: ", time.time() - strat_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 3: 整型数组和零维元素")
seq = [np.array([1, 2, 3], dtype=np.int64), np.array(42, dtype=np.int64), np.array([[5]], dtype=np.int32)]
pos = 1

np_out = onnx_sequence_at(seq, pos)
kp_out = sequence_at_op.run(seq, np.array([[pos]], dtype=np.int64))[0]
print(f"{sequence_at_op}: ", time.time() - strat_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

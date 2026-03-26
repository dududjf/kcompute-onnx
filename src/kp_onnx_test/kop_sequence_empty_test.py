import time
import numpy as np
from kp import Manager
from kp_onnx_ssbo.kop_sequence_empty import SequenceEmptyOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sequence_empty_op = SequenceEmptyOp(mgr)


def onnx_sequence_empty(dtype=None):
    return []


print("Case")
strat_time = time.time()
np_out = onnx_sequence_empty()
print("Numpy: ", time.time() - strat_time, "seconds")

strat_time = time.time()
kp_out = sequence_empty_op.run()[0]
print(f"{sequence_empty_op}: ", time.time() - strat_time, "seconds")

print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')
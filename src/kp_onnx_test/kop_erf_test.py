from kp import Manager
import numpy as np
import time
from math import erf
from kp_onnx.kop_erf import ErfOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

erf_op = ErfOp(mgr)
vec_erf_func = np.vectorize(erf)
numpy_in = np.random.random(1024 * 1024)

start_time = time.time()
numpy_out = vec_erf_func(numpy_in)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = erf_op.run(numpy_in)[0]
print(f"{erf_op}:", time.time() - start_time, "seconds")

print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

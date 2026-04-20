from kp import Manager
import numpy as np
import time
from kp_onnx.kop_tile import TileOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

tile_op = TileOp(mgr)

print("Case 1")
x = np.random.random((3, 2, 63, 511)).astype(np.float32)
repeats = np.array([2, 2, 3, 4], dtype=np.int64)

start_time = time.time()
np_out = np.tile(x, repeats)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = tile_op.run(x, repeats)[0]
print(f"{tile_op}:", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

print("Case 2")
x = np.random.random((3, 2, 63, 511)).astype(np.float32)
repeats = [2, 1, 3, 1]

start_time = time.time()
np_out = np.tile(x, repeats)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = tile_op.run(x, repeats)[0]
print(f"{tile_op}:", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

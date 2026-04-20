import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_depth_to_space import DepthToSpaceOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

depth_to_space_op = DepthToSpaceOp(mgr)
def numpy_depth_to_space(data, blocksize=None, mode=None):
    if len(data.shape) != 4:
        raise RuntimeError(f"Unexpected shape {data.shape!r}.")
    b, c, h, w = data.shape
    if mode == "DCR":
        tmpshape = (
            b,
            blocksize,
            blocksize,
            c // (blocksize * blocksize),
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 3, 4, 1, 5, 2])
    else:
        # assert mode == "CRD"
        tmpshape = (
            b,
            c // (blocksize * blocksize),
            blocksize,
            blocksize,
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 1, 4, 2, 5, 3])
    finalshape = (
        b,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    y = np.reshape(transposed, finalshape)
    return y


x = np.random.random((2, 16, 512, 512)).astype(np.float32)

print("Case 1: mode is 'DCR', blocksize is 4")
start_time = time.time()
np_out = numpy_depth_to_space(x, blocksize=4, mode="DCR")
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
depth_to_space_op.blocksize = 4
depth_to_space_op.mode = "DCR"
kp_out = depth_to_space_op.run(x)[0]
print(f"{depth_to_space_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 2: mode is 'CRD', blocksize is 4")
start_time = time.time()
np_out = numpy_depth_to_space(x, blocksize=4, mode="CRD")
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
depth_to_space_op.blocksize = 4
depth_to_space_op.mode = "CRD"
kp_out = depth_to_space_op.run(x)[0]
print(f"{depth_to_space_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
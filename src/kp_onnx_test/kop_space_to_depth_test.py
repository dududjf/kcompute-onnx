import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_space_to_depth import SpaceToDepthOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

space_to_depth_op = SpaceToDepthOp(mgr)


def numpy_space_to_depth(data, blocksize=None):  # type: ignore
    if len(data.shape) != 4:
        raise RuntimeError(f"Unexpected shape {data.shape!r}.")
    b, C, H, W = data.shape
    tmpshape = (
        b,
        C,
        H // blocksize,
        blocksize,
        W // blocksize,
        blocksize,
    )
    reshaped = np.reshape(data, tmpshape)
    transposed = np.transpose(reshaped, [0, 3, 5, 1, 2, 4])
    finalshape = (
        b,
        C * blocksize * blocksize,
        H // blocksize,
        W // blocksize,
    )
    y = np.reshape(transposed, finalshape).astype(data.dtype)
    return y


x = np.random.random((32, 16, 512, 512)).astype(np.float32)

print("Case: blocksize is 4")
start_time = time.time()
np_out = numpy_space_to_depth(x, blocksize=4)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
space_to_depth_op.blocksize = 4
kp_out = space_to_depth_op.run(x)[0]
print(f"{space_to_depth_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
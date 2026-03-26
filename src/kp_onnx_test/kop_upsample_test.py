from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_upsample import UpsampleOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

upsample_op = UpsampleOp(mgr)


def np_upsample(x, scale, mode=None):
    if mode == "nearest" and scale.astype(np.int64).tolist() == scale.tolist():
        r = x
        for axis, s in enumerate(scale):
            if s == 1:
                continue
            r = np.repeat(r, int(s), axis=axis)
        return r
    raise RuntimeError(f"Not implemented for mode={mode!r} and scale={scale!r}.")


print("Case: ")
x = np.random.random((32, 2, 63, 512)).astype(np.float32)
scales = np.array([2.0, 3.0, 1.0, 2.0], dtype=np.float32)

start_time = time.time()
np_out = np_upsample(x, scales, "nearest")
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
upsample_op.mode = "nearest"
kp_out = upsample_op.run(x, scales)[0]
print(f"{upsample_op}: ", kp_out.shape, time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')
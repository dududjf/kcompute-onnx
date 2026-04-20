from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_sigmoid import SigmoidOp


def sigmoid(x):
    if x > 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


class NumpySigmoid:
    def __init__(self):
        self.vf = np.vectorize(sigmoid)

    def run(self, X: np.ndarray):
        if len(X.shape) == 0:
            return (sigmoid(X).astype(X.dtype),)
        if X.size == 0:
            return (X,)
        return (self.vf(X).astype(X.dtype),)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

sigmoid_op = SigmoidOp(mgr)
numpy_sigmoid = NumpySigmoid()
numpy_in = np.random.uniform(-1000, 1000, 1024 * 1024 * 16).astype(np.float32)

start_time = time.time()
numpy_out = numpy_sigmoid.run(numpy_in)[0]
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = sigmoid_op.run(numpy_in)[0]
print(f"{sigmoid_op}:", time.time() - start_time, "seconds")

print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

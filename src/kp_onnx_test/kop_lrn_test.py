from kp import Manager
import numpy as np
import time
from kp_onnx.kop_lrn import LRNOp


def np_lrn(x, alpha, beta, bias, size):
    if len(x.shape) != 4:
        raise RuntimeError(f"LRN only applies on 4D tensors but got shape={x.shape}")

    square_sum = np.zeros(x.shape).astype(x.dtype)
    C = x.shape[1]

    c1 = int(np.floor((size - 1) / 2))
    c2 = int(np.ceil((size - 1) / 2)) + 1

    for c in range(C):
        begin = max(0, c - c1)
        end = min(C, c + c2)
        square_sum[:, c, :, :] = np.sum(x[:, begin:end, :, :] ** 2, axis=1)

    y = x / ((bias + (alpha / size) * square_sum) ** beta)
    return y.astype(x.dtype)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
lrn_op = LRNOp(mgr)


print('Case 1')
numpy_in = np.random.random((4, 32, 255, 255)).astype(np.float32)
lrn_op.alpha = 1e-4
lrn_op.beta = 0.75
lrn_op.bias = 1.0
lrn_op.size = 5
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 2')
numpy_in = np.random.random((2, 128, 128, 128)).astype(np.float32)
lrn_op.alpha = 2e-4
lrn_op.beta = 0.5
lrn_op.bias = 1.0
lrn_op.size = 3
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 3')
numpy_in = np.random.random((64, 32, 64, 64)).astype(np.float32)
lrn_op.alpha = 1e-4
lrn_op.beta = 0.75
lrn_op.bias = 1.0
lrn_op.size = 7
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 4')
numpy_in = np.random.random((4, 16, 256, 512)).astype(np.float32)
lrn_op.alpha = 5e-5
lrn_op.beta = 1.0
lrn_op.bias = 1.0
lrn_op.size = 5
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 5')
numpy_in = np.random.random((8, 3, 1024, 512)).astype(np.float32)
lrn_op.alpha = 1e-4
lrn_op.beta = 0.5
lrn_op.bias = 1.0
lrn_op.size = 5
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 6')
numpy_in = np.random.random((4, 32, 256, 256)).astype(np.float32)
lrn_op.alpha = 1e-4
lrn_op.beta = 0.75
lrn_op.bias = 1.0
lrn_op.size = 1
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 7')
numpy_in = np.random.random((2048, 2048, 1, 1)).astype(np.float32)
lrn_op.alpha = 1e-4
lrn_op.beta = 1.0
lrn_op.bias = 2.0
lrn_op.size = 5
start_time = time.time()
np_out = np_lrn(numpy_in, lrn_op.alpha, lrn_op.beta, lrn_op.bias, lrn_op.size)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = lrn_op.run(numpy_in)[0]
print(f"{lrn_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

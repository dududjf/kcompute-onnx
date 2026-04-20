from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_matmul_integer import MatMulIntegerOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
matmul_integer_op = MatMulIntegerOp(mgr)


def np_matmul_integer(A, B, a_zero_point=None, b_zero_point=None):  # type: ignore
    A32 = A.astype(np.int32)
    if a_zero_point is not None:
        A32 -= a_zero_point
    B32 = B.astype(np.int32)
    if b_zero_point is not None:
        B32 -= b_zero_point
    return A32 @ B32


print('Case 1')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}: ", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 2')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.uint8)
numpy_in_2 = np.random.random((2, 5, 512, 1024)).astype(np.uint8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 3: 2D B with azp (per-row) and bzp (per-col)')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)
a_zero_point = np.random.random((2, 5, 1000, 1)).astype(np.int8)
b_zero_point = np.random.random((1024,)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 4: batched with azp (per-batch-row) and bzp (per-batch-col)')
numpy_in_1 = np.random.random((2, 5, 1000, 512)).astype(np.uint8)
numpy_in_2 = np.random.random((2, 5, 512, 1024)).astype(np.uint8)
a_zero_point = np.random.random((2, 5, 1000, 1)).astype(np.uint8)
b_zero_point = np.random.random((2, 5, 1, 1024)).astype(np.uint8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 5: 2D with azp (per-row) and bzp (per-col)')
numpy_in_1 = np.random.random((1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)
a_zero_point = np.random.random((1000,)).astype(np.int8)
b_zero_point = np.random.random((1024,)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))

print('Case 6: 2D with scalar azp and bzp (mode=0)')
numpy_in_1 = np.random.random((1000, 512)).astype(np.int8)
numpy_in_2 = np.random.random((512, 1024)).astype(np.int8)
a_zero_point = np.array(2, dtype=np.int8)
b_zero_point = np.array(5, dtype=np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 7: batched with scalar azp and bzp (mode=0)')
numpy_in_1 = np.random.random((2, 3, 100, 128)).astype(np.int8)
numpy_in_2 = np.random.random((2, 3, 128, 256)).astype(np.int8)
a_zero_point = np.array(3, dtype=np.int8)
b_zero_point = np.array(7, dtype=np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print('Case 8: batched with azp (per-row, mode=1) and bzp (per-col, mode=1)')
numpy_in_1 = np.random.random((2, 3, 100, 128)).astype(np.int8)
numpy_in_2 = np.random.random((2, 3, 128, 256)).astype(np.int8)
a_zero_point = np.random.random((100,)).astype(np.int8)
b_zero_point = np.random.random((256,)).astype(np.int8)

start_time = time.time()
numpy_out = np_matmul_integer(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = matmul_integer_op.run(numpy_in_1, numpy_in_2, a_zero_point, b_zero_point)[0]
print(f"{matmul_integer_op}:", time.time() - start_time, "seconds")

print('Max error:', np.abs(numpy_out - kp_out).max())
print(np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
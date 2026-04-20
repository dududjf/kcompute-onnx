from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_mod import ModOp


def onnx_mod(a, b, fmod=0):
    if fmod == 1:
        result = np.fmod(a, b)
    elif a.dtype in (np.float16, np.float32, np.float64):
        result = np.nan_to_num(np.fmod(a, b))
    else:
        result = np.nan_to_num(np.mod(a, b))
    return result


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
mod_op = ModOp(mgr)

print('Case 1')
numpy_in_1 = np.random.random((128, 15))
numpy_in_2 = np.random.random((3, 64, 128, 1))

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)   # baseline
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())

print('Case 2')
numpy_in_1 = np.random.random((3, 64, 64, 1))
numpy_in_2 = np.random.random((64, 15))

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())

print('Case 3')
numpy_in_1 = np.random.random((64, 64, 5))
numpy_in_2 = np.random.random((3, 3, 64, 64, 1))

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())

print('Case 4')
numpy_in_1 = np.random.random((3, 3, 64, 64, 1))
numpy_in_2 = np.random.random((64, 64, 15))

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())

print('Case 5')
numpy_in_1 = np.random.random((1023, 15))
numpy_in_2 = np.random.random((1023, 1))

start_time = time.time()
numpy_out = np.fmod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())

print('Case 6')
numpy_in_1 = np.random.random((1023,))
numpy_in_2 = np.random.random((1,))

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())

print('Case 7: 输入A、B张量都是int32, fmod=0')
numpy_in_1 = np.random.random((1023,)).astype(np.int32)
numpy_in_2 = np.random.random((1,)).astype(np.int32)

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
mod_op.fmod = False
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("kp_out.atype:", kp_out[0].dtype)

print('Case 8: 输入A、B张量都是int32，fmod=True')
numpy_in_1 = np.random.random((1023,)).astype(np.int32)
numpy_in_2 = np.random.random((1,)).astype(np.int32)

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2, fmod=1)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
mod_op.fmod = True
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4, equal_nan=True))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("kp_out.atype:", kp_out[0].dtype)

print('Case 9: 输入A为float32，输入B为int32')
numpy_in_1 = np.random.random((1023,)).astype(np.float32)
numpy_in_2 = np.random.random((1,)).astype(np.int32)

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
mod_op.fmod = False
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4, equal_nan=True))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("kp_out.atype:", kp_out[0].dtype)

print('Case 10: 输入A为int32，输入B为float32')
numpy_in_1 = np.random.random((1,)).astype(np.int32)
numpy_in_2 = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
mod_op.fmod = False
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("kp_out.atype:", kp_out[0].dtype)

print('Case 11: 输入A为int32，输入B为float32, fmod=1')
numpy_in_1 = np.random.random((1023,)).astype(np.int32)
numpy_in_2 = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2, fmod=1)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
mod_op.fmod = True
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("kp_out.atype:", kp_out[0].dtype)

print('Case 12: 输入A、B张量都是float32')
numpy_in_1 = np.random.random((1,)).astype(np.float32)
numpy_in_2 = np.random.random((1023,)).astype(np.float32)

start_time = time.time()
numpy_out = onnx_mod(numpy_in_1, numpy_in_2)
print("Numpy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
mod_op.fmod = False
kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
print(f"{mod_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("kp_out.atype:", kp_out[0].dtype)

print('Case 13: 测试不支持的数据类型uint8，应该抛出TypeError异常')
numpy_in_1 = np.random.randint(0, 255, size=(10,), dtype=np.uint8)
numpy_in_2 = np.random.randint(1, 10, size=(10,), dtype=np.uint8)

try:
    kp_out = mod_op.run(numpy_in_1, numpy_in_2)[0]
    print("错误：应该抛出TypeError异常，但没有抛出")
except TypeError as e:
    print(f"成功捕获TypeError异常: {e}")
    print("测试通过：正确识别了不支持的数据类型")

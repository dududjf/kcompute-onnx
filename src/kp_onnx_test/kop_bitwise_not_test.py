from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_bitwise_not import BitwiseNotOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

bitwise_not_op = BitwiseNotOp(mgr)

def numpy_bitwise_not_float(x):
    return np.bitwise_not(x.view(np.int32)).view(x.dtype)

#float32
numpy_in_f32 = np.random.uniform(-1000, 1000, 1024 * 1024).astype(np.float32)

start_time = time.time()
numpy_out_f32 = numpy_bitwise_not_float(numpy_in_f32)
print("Numpy float32:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out_f32 = bitwise_not_op.run(numpy_in_f32)[0]
print(f"{bitwise_not_op} float32:", time.time() - start_time, "seconds")

print("float32 match:", np.allclose(numpy_out_f32, kp_out_f32, rtol=1e-4, atol=1e-4))


#int32
numpy_in_i32 = np.random.randint(-1000, 1000, 1024 * 1024)

start_time = time.time()
numpy_out_i32 = np.bitwise_not(numpy_in_i32)
print("Numpy int32:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out_i32 = bitwise_not_op.run(numpy_in_i32)[0]
print(f"{bitwise_not_op} int32:", time.time() - start_time, "seconds")

print("int32 match:", np.allclose(numpy_out_i32, kp_out_i32, rtol=1e-4, atol=1e-4))

from kp import Manager
import numpy as np
import time
from kp_onnx.kop_gather_elements import GatherElementsOp


def gather_numpy_2(self: np.ndarray, index: np.ndarray) -> np.ndarray:
    res = []
    for a, b in zip(self, index):
        res.append(a[b[0]])
    return np.array(res, dtype=self.dtype).reshape(index.shape)


def gather_numpy(self: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(
            f"Except for dimension {dim!r}, all dimensions of "
            f"index and self should be the same size."
        )
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    try:
        gathered = np.choose(index_swaped, data_swaped, mode="wrap")
    except ValueError:
        if len(index_swaped.shape) == 2 and len(data_swaped.shape) == 2:
            return gather_numpy_2(self, index)
        raise
    return np.swapaxes(gathered, 0, dim)


def np_gather_elements(data, indices, axis):
    if indices.size == 0:
        return np.empty((0,), dtype=data.dtype)
    try:
        return gather_numpy(data, axis, indices)
    except TypeError:
        return gather_numpy(data, axis, indices.astype(int))


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())
gather_op = GatherElementsOp(mgr)

# Case 1: 基础二维测试（axis=1）
print('Case 1')
numpy_in = np.random.random((1023, 15)).astype(np.float32)
indices_in = np.random.randint(0, 15, (1023, 15)).astype(np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 1
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 2: 一维输入与一维索引
print('Case 2')
numpy_in = np.random.random((31,)).astype(np.float32)
indices_in = np.random.randint(0, 31, (31,)).astype(np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=0)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 0
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 3: 三维输入 + 三维索引（负数索引）
print('Case 3')
numpy_in = np.random.random((225, 255, 15)).astype(np.float32)
indices_in = np.random.randint(-15, 15, (225, 255, 15)).astype(np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=2)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 2
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 4: 标量索引（单值）
print('Case 4')
numpy_in = np.random.random((31,)).astype(np.float32)   # 1D, length <=32
indices_in = np.random.randint(0, 31, (16,)).astype(np.int32)  # output length 16

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=0)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 0
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 5: 二维输入（axis=0）
print('Case 5')
numpy_in = np.random.random((31, 2048)).astype(np.float32)
indices_in = np.random.randint(0, 31, (1023, 2048)).astype(np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=0)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 0
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 6: 三维输入（axis=-1）
print('Case 6')
numpy_in = np.random.random((64, 64, 31)).astype(np.float32)
indices_in = np.random.randint(0, 31, (64, 64, 31)).astype(np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=-1)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = -1
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 7: 重复索引
print('Case 7')
numpy_in = np.arange(16).astype(np.float32)
indices_in = np.array([0,0,15,15,7,7,3,3,1,1,5,5,2,2,6,6], dtype=np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=0)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 0
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

# Case 8: 四维输入测试
print('Case 8')
numpy_in = np.random.random((3, 1023, 15, 1023)).astype(np.float32)
indices_in = np.random.randint(0, 15, (3, 1023, 15, 1023)).astype(np.int32)

start_time = time.time()
np_out = np_gather_elements(numpy_in, indices_in, axis=2)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

gather_op.axis = 2
start_time = time.time()
kp_out = gather_op.run(numpy_in, indices_in)[0]
print(f"{gather_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-5, atol=1e-5))

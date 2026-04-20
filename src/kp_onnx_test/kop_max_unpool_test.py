from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_max_unpool import MaxUnpoolOp  # 使用正式实现


def onnx_reference_max_unpool(X, indices, kernel_shape, strides=None, pads=None, output_shape=None):
    """参考实现：基于ONNX官方示例的MaxUnpool"""
    X = np.array(X, dtype=np.float32)
    indices = np.array(indices, dtype=np.int64)

    pooling_dims = len(X.shape) - 2
    if pooling_dims > 3:
        raise NotImplementedError(f"Unsupported pooling size {pooling_dims}")

    strides = strides or [1] * pooling_dims
    pads = pads or [0] * (2 * pooling_dims)

    # 计算推断形状
    inferred_shape = list(X.shape)
    for dim in range(pooling_dims):
        inferred_shape[dim + 2] = (
                (X.shape[dim + 2] - 1) * strides[dim]
                - (pads[dim] + pads[pooling_dims + dim])
                + kernel_shape[dim]
        )

    # 确定输出形状
    if output_shape is None:
        shape = inferred_shape
    else:
        shape = output_shape

    # 执行反池化操作
    total_elements = np.prod(X.shape)
    Y_flat = np.zeros(np.prod(inferred_shape), dtype=X.dtype)

    I_flat = indices.flatten()
    X_flat = X.flatten()

    for i in range(total_elements):
        if 0 <= I_flat[i] < len(Y_flat):
            Y_flat[I_flat[i]] = X_flat[i]

    Y = Y_flat.reshape(tuple(inferred_shape))

    # 处理指定输出形状（可能大于推断形状）
    res = np.zeros(shape, dtype=Y.dtype)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(inferred_shape, shape))
    res[slices] = Y[slices]

    return res


# 生成测试用的池化索引（模拟MaxPool过程）
def generate_max_indices(input_tensor, kernel_shape, strides, pads):
    """辅助函数：生成MaxPool的索引用于测试MaxUnpool"""
    input_shape = input_tensor.shape
    n, c = input_shape[0], input_shape[1]
    spatial_dims = len(input_shape) - 2
    kernel = list(kernel_shape)
    strides = strides or [1] * spatial_dims
    pads = pads or [0] * (2 * spatial_dims)

    # 计算输出形状
    output_spatial = []
    for dim in range(spatial_dims):
        out_dim = (input_shape[dim + 2] + pads[dim] + pads[dim + spatial_dims] - kernel[dim]) // strides[dim] + 1
        output_spatial.append(out_dim)
    output_shape = (n, c) + tuple(output_spatial)

    # 初始化输出和索引
    output = np.zeros(output_shape, dtype=input_tensor.dtype)
    indices = np.zeros(output_shape, dtype=np.int64)

    # 填充输入（添加padding）
    pad_width = [(0, 0), (0, 0)]
    for dim in range(spatial_dims):
        pad_width.append((pads[dim], pads[dim + spatial_dims]))
    padded = np.pad(input_tensor, pad_width, mode='constant')

    # 计算扁平化索引偏移量
    spatial_sizes = input_shape[2:]
    flat_strides = [1] * spatial_dims
    for i in range(spatial_dims - 2, -1, -1):
        flat_strides[i] = flat_strides[i + 1] * spatial_sizes[i + 1]
    flat_strides = [n * c * np.prod(spatial_sizes)] + [np.prod(spatial_sizes)] + flat_strides

    # 执行MaxPool并记录索引
    for batch in range(n):
        for channel in range(c):
            # 生成空间维度的网格 - 将列表转换为元组
            for idx in np.ndindex(tuple(output_spatial)):
                # 计算窗口在输入中的位置
                window_pos = []
                for dim in range(spatial_dims):
                    start = idx[dim] * strides[dim]
                    end = start + kernel[dim]
                    window_pos.append(slice(start, end))

                # 提取窗口并找到最大值
                window = padded[batch, channel][tuple(window_pos)]
                max_val = np.max(window)
                max_pos = np.unravel_index(np.argmax(window), window.shape)

                # 计算原始输入中的坐标（去除padding）
                original_pos = []
                for dim in range(spatial_dims):
                    pos = idx[dim] * strides[dim] + max_pos[dim] - pads[dim]
                    original_pos.append(pos)

                # 验证坐标是否在原始输入范围内
                valid_coords = True
                for dim in range(spatial_dims):
                    if not (0 <= original_pos[dim] < input_shape[dim + 2]):
                        valid_coords = False
                        break

                if not valid_coords:
                    # 如果坐标无效，设置为-1或0
                    flat_idx = -1
                else:
                    # 计算扁平化索引
                    flat_idx = batch * flat_strides[0] + channel * flat_strides[1]
                    for dim in range(spatial_dims):
                        flat_idx += original_pos[dim] * flat_strides[2 + dim]

                # 保存结果
                output[batch, channel][idx] = max_val
                indices[batch, channel][idx] = flat_idx

    return output, indices


# 测试执行部分
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

unpool_op = MaxUnpoolOp(mgr)

# Case 1: 1D, 使用推断形状
print("Case 1: 1D, using inferred shape")
np.random.seed(42)
input_shape = (2, 3, 1024)  # N, C, W
kernel_shape = [3]
strides = [2]
pads = [1, 1]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 1D, 使用指定output_shape
print("Case 2: 1D, using specified output_shape")
output_shape = (2, 3, 1024)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides,
    pads=pads, output_shape=output_shape
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices, output_shape)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 1D, stride > kernel
print("Case 3: 1D, stride > kernel")
input_shape = (2, 3, 1024)
kernel_shape = [2]
strides = [3]  # stride > kernel
pads = [0, 0]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 1D, 单元素输出
print("Case 4: 1D, single element output")
input_shape = (2, 3, 5)
kernel_shape = [5]
strides = [1]
pads = [0, 0]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: 2D, 使用推断形状
print("Case 5: 2D, using inferred shape")
input_shape = (2, 3, 512, 512)  # N, C, H, W
kernel_shape = [3, 3]
strides = [2, 2]
pads = [1, 1, 1, 1]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: 2D, 使用指定output_shape
print("Case 6: 2D, using specified output_shape")
output_shape = (2, 3, 512, 512)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides,
    pads=pads, output_shape=output_shape
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices, output_shape)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: 2D, 不对称padding
print("Case 7: 2D, asymmetric padding")
input_shape = (2, 3, 512, 512)
kernel_shape = [3, 3]
strides = [2, 2]
pads = [0, 1, 2, 3]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 8: 2D, 带无效索引
print("Case 8: 2D, with invalid indices")
input_shape = (2, 3, 512, 512)
kernel_shape = [3, 3]
strides = [2, 2]
pads = [1, 1, 1, 1]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

# 手动设置一些无效索引
indices[0, 0, 0, 0] = -1  # 负索引
indices[1, 2, 2, 2] = 100000  # 超出范围的索引

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 9: 2D, 输出形状小于推断形状
print("Case 9: 2D, output_shape smaller than inferred")
input_shape = (2, 3, 512, 512)
kernel_shape = [3, 3]
strides = [2, 2]
pads = [1, 1, 1, 1]
output_shape = (2, 3, 8, 8)  # 小于推断形状

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides,
    pads=pads, output_shape=output_shape
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices, output_shape)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 10: 2D, 无padding
print("Case 10: 2D, no padding")
input_shape = (2, 3, 512, 512)
kernel_shape = [3, 3]
strides = [2, 2]
pads = [0, 0, 0, 0]  # 无padding

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 11: 3D, 使用推断形状
print("Case 11: 3D, using inferred shape")
input_shape = (2, 3, 100, 100, 100)  # N, C, D, H, W
kernel_shape = [2, 2, 2]
strides = [1, 1, 1]
pads = [0, 0, 0, 0, 0, 0]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 12: 3D, 使用指定output_shape
print("Case 12: 3D, using specified output_shape")
output_shape = (2, 3, 100, 100, 100)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides,
    pads=pads, output_shape=output_shape
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices, output_shape)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 13: 3D, 大kernel
print("Case 13: 3D, large kernel")
input_shape = (2, 3, 100, 100, 100)
kernel_shape = [5, 5, 5]
strides = [1, 1, 1]
pads = [2, 2, 2, 2, 2, 2]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 14: 3D, 与输入形状相同
print("Case 14: 3D, same as input shape")
input_shape = (2, 3, 100, 100, 100)
kernel_shape = [1, 1, 1]
strides = [1, 1, 1]
pads = [0, 0, 0, 0, 0, 0]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides, pads=pads
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 15: 2D，output_shape 只包含空间维度 (H, W)
print("Case 15: 2D, output_shape contains only spatial dims (H, W) — triggers line 128-131")
np.random.seed(42)
input_shape = (2, 3, 64, 64)
kernel_shape = [3, 3]
strides = [2, 2]
pads = [1, 1, 1, 1]

input_tensor = np.random.uniform(-8, 8, input_shape).astype(np.float32)
X, indices = generate_max_indices(input_tensor, kernel_shape, strides, pads)

# 推断形状是 (2, 3, 128, 128)，只传空间部分 (128, 128)
output_shape_spatial = np.array([128, 128], dtype=np.int64)
full_output_shape_15 = (2, 3, 128, 128)

start_time = time.time()
numpy_out = onnx_reference_max_unpool(
    X, indices, kernel_shape=kernel_shape, strides=strides,
    pads=pads, output_shape=full_output_shape_15
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
unpool_op.kernel_shape = kernel_shape
unpool_op.strides = strides
unpool_op.pads = pads
kp_out = unpool_op.run(X, indices, output_shape_spatial)[0]
print(f"{unpool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

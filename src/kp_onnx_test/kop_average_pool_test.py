from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_average_pool import AveragePoolOp


def onnx_reference_average_pool(x, kernel_shape, strides, pads=None, auto_pad="", ceil_mode=0, count_include_pad=0, dilations=None):
    x_shape = list(np.shape(x))
    pooling_type = "AVG"
    pading_value = np.nan if pooling_type == "MAX" or count_include_pad == 0 else 0

    if auto_pad in ["SAME_UPPER", "SAME_LOWER", "VALID"]:
        assert ceil_mode == 0, (
            "ceil_mode is not supported with auto_pad"
        )
        out_shape = get_output_shape_auto_pad(
            auto_pad, list(x.shape[2:]), kernel_shape, strides, dilations
        )
        pads_shape = get_pad_shape(
            auto_pad, x_shape[2:], kernel_shape, strides, out_shape
        )
        pads = get_pad_with_auto_pad(auto_pad, pads_shape)
        n_dims = len(pads) // 2
        pads_np = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
        padded = np.pad(
            x,
            ((0, 0), (0, 0), *pads_np),
            mode="constant",
            constant_values=pading_value,
        )
        y = pool(
            padded,
            x_shape,
            kernel_shape,
            strides,
            out_shape,
            pooling_type,
            pads,
            pads,
            dilations,
            count_include_pad,
            1, # p
        )
        return y
    else:
        out_shape, extra_pads = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides, dilations, ceil_mode
        )
        # convert pads from [x1_begin, x2_begin,...,x1_end, x2_end,...] to [(x1_begin, x1_end), (x2_begin, x2_end),...]
        n_dims = len(extra_pads) // 2
        pads_np = [(extra_pads[i], extra_pads[i + n_dims]) for i in range(n_dims)]
        padded = np.pad(
            x,
            ((0, 0), (0, 0), *pads_np),
            mode="constant",
            constant_values=pading_value,
        )
        y = pool(
            padded,
            x_shape,
            kernel_shape,
            strides,
            out_shape,
            pooling_type,
            extra_pads,
            pads,
            dilations,
            count_include_pad,
            1, # p
        )
        return y


def get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
    spatial_dims = len(input_spatial_shape)
    pad_shape = [0] * spatial_dims
    strides_spatial = strides_spatial or [1] * spatial_dims
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(spatial_dims):
            pad_shape[i] = (
                (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i]
            )
    elif auto_pad == "VALID":
        pass
    return pad_shape


def get_pad_with_auto_pad(auto_pad, pad_shape):
    spatial_dims = len(pad_shape)
    if auto_pad == "SAME_UPPER":
        pads = [pad_shape[i] // 2 for i in range(spatial_dims)] + [
            pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)
        ]
    elif auto_pad == "SAME_LOWER":
        pads = [pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)] + [
            pad_shape[i] // 2 for i in range(spatial_dims)
        ]
    else:
        pads = [0] * spatial_dims * 2  # no padding
    return pads


def get_output_shape_explicit_padding(pads, input_spatial_shape, kernel_spatial_shape, strides_spatial, dilations=None, ceil_mode=False):
    output_spatial_shape = [0] * len(input_spatial_shape)
    pads = list(pads) if pads is not None else [0] * len(input_spatial_shape) * 2
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    dims = len(input_spatial_shape)
    if dilations is None:
        dilations = [1] * dims

    for dim in range(dims):
        dim_size = (
            input_spatial_shape[dim]
            + pads[dim]
            + pads[dims + dim]
            - dilations[dim] * (kernel_spatial_shape[dim] - 1)
            - 1
        ) / strides_spatial[dim] + 1

        if ceil_mode:
            output_spatial_shape[dim] = int(np.ceil(dim_size))
            if (output_spatial_shape[dim] - 1) * strides_spatial[dim] >= input_spatial_shape[dim] + pads[dim]:
                output_spatial_shape[dim] -= 1
        else:
            output_spatial_shape[dim] = int(np.floor(dim_size))

    pads_spatial_shape_new = pads[:]
    for dim in range(dims):
        sliding_window_size = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
        actual_padded_input_size = (output_spatial_shape[dim] - 1) * strides_spatial[dim] + sliding_window_size
        extra_pad = (
            actual_padded_input_size
            - input_spatial_shape[dim]
            - pads[dim]
            - pads[dims + dim]
        )
        if extra_pad > 0:
            pads_spatial_shape_new[dim] += extra_pad // 2
            pads_spatial_shape_new[dims + dim] += extra_pad - extra_pad // 2

    return output_spatial_shape, pads_spatial_shape_new


def get_output_shape_auto_pad(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, dilations=None):
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    dilations = dilations or [1] * len(input_spatial_shape)
    out_shape = [0] * len(input_spatial_shape)

    for i in range(len(input_spatial_shape)):
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out_shape[i] = (
                    int(np.floor((input_spatial_shape[i] - 1) / strides_spatial[i])) + 1
            )
        elif auto_pad == "VALID":
            effective_kernel_size = (kernel_spatial_shape[i] - 1) * dilations[i] + 1
            out_shape[i] = (
                    int(np.floor(
                        (input_spatial_shape[i] - effective_kernel_size) / strides_spatial[i]
                    ))
                    + 1
            )
        else:
            raise ValueError("auto_pad can only be NOTSET, SAME_UPPER, SAME_LOWER, or VALID")
    return out_shape


def pool(padded, x_shape, kernel, strides, out_shape, pooling_type, pads_required=None, pads=None, dilations=None, count_include_pad=0, p=1):
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1], *list(out_shape)], dtype=padded.dtype)
    if dilations is None:
        dilations = [1] * spatial_size
    if pads_required is None:
        pads_required = [0] * (spatial_size * 2)
    if pads is None:
        pads = [0] * (spatial_size * 2)
    strides = strides or [1] * spatial_size

    import itertools
    for shape in itertools.product(
        range(x_shape[0]),
        range(x_shape[1]),
        *[
            range(
                int(
                    (
                        x_shape[i + 2]
                        + pads_required[i]
                        + pads_required[i + spatial_size]
                        - (1 + (kernel[i] - 1) * dilations[i])
                    )
                    / strides[i]
                    + 1
                )
            )
            for i in range(spatial_size)
        ],
    ):
        window = padded[shape[0], shape[1]]
        window_vals = np.array(
            [
                window[i]
                for i in list(
                    itertools.product(
                        *[
                            [
                                pixel
                                for pixel in range(
                                    strides[i] * shape[i + 2],
                                    strides[i] * shape[i + 2]
                                    + (1 + (kernel[i] - 1) * dilations[i]),
                                    dilations[i],
                                )
                                if pixel
                                < x_shape[i + 2] + pads[i] + pads[spatial_size + i]
                            ]
                            for i in range(spatial_size)
                        ]
                    )
                )
            ]
        )

        if pooling_type == "AVG":
            f = np.average
        elif pooling_type == "MAX":
            f = np.max
        elif pooling_type == "LPPOOL":
            def lp_pool(x: np.array, p: int = p) -> float:
                return np.sum(np.abs(x) ** p) **(1.0 / p)
            f = lp_pool
        else:
            raise NotImplementedError(f"Pooling type {pooling_type} not supported")

        if count_include_pad == 1 and (pooling_type in {"AVG", "LPPOOL"}):
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(padded.dtype)


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

pool_op = AveragePoolOp(mgr)

# Case 1: 1D, auto_pad=SAME_UPPER, count_include_pad=0
print("Case 1: 1D, auto_pad=SAME_UPPER, count_include_pad=0")
numpy_in = np.random.uniform(-8, 8, (2, 3, 4096)).astype(np.float32)  # NCHW格式: (N,C,D)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3], strides=[2], auto_pad="SAME_UPPER"
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3]
pool_op.strides = [2]
pool_op.auto_pad = "SAME_UPPER"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 1D, auto_pad=SAME_LOWER, count_include_pad=1
print("Case 2: 1D, auto_pad=SAME_LOWER, count_include_pad=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3], strides=[2], auto_pad="SAME_LOWER", count_include_pad=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3]
pool_op.strides = [2]
pool_op.auto_pad = "SAME_LOWER"
pool_op.count_include_pad = 1
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 1D, auto_pad=VALID, with dilations
print("Case 3: 1D, auto_pad=VALID, with dilations")
numpy_in = np.random.uniform(-8, 8, (2, 3, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3], strides=[2], auto_pad="VALID", dilations=[2]
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3]
pool_op.strides = [2]
pool_op.auto_pad = "VALID"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = [2]
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 1D, explicit padding, ceil_mode=1
print("Case 4: 1D, explicit padding, ceil_mode=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3], strides=[2], pads=[1,1], ceil_mode=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3]
pool_op.strides = [2]
pool_op.auto_pad = ""
pool_op.count_include_pad = 0
pool_op.ceil_mode = 1
pool_op.dilations = None
pool_op.pads = [1,1]
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: 1D, large kernel, same padding
print("Case 5: 1D, large kernel, same padding")
numpy_in = np.random.uniform(-8, 8, (2, 3, 4096)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[5], strides=[1], auto_pad="SAME_UPPER"
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [5]
pool_op.strides = [1]
pool_op.auto_pad = "SAME_UPPER"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: 2D, auto_pad=SAME_UPPER, count_include_pad=0
print("Case 6: 2D, auto_pad=SAME_UPPER, count_include_pad=0")
numpy_in = np.random.uniform(-8, 8, (2, 3, 1024, 1024)).astype(np.float32)  # NCHW
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3], strides=[2,2], auto_pad="SAME_UPPER"
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3]
pool_op.strides = [2,2]
pool_op.auto_pad = "SAME_UPPER"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: 2D, auto_pad=SAME_LOWER, count_include_pad=1
print("Case 7: 2D, auto_pad=SAME_LOWER, count_include_pad=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 1024, 1024)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3], strides=[2,2], auto_pad="SAME_LOWER", count_include_pad=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3]
pool_op.strides = [2,2]
pool_op.auto_pad = "SAME_LOWER"
pool_op.count_include_pad = 1
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 8: 2D, auto_pad=VALID, with dilations
print("Case 8: 2D, auto_pad=VALID, with dilations")
numpy_in = np.random.uniform(-8, 8, (2, 3, 1024, 1024)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3], strides=[2,2], auto_pad="VALID", dilations=[2,2]
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3]
pool_op.strides = [2,2]
pool_op.auto_pad = "VALID"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = [2,2]
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 9: 2D, explicit padding, ceil_mode=1
print("Case 9: 2D, explicit padding, ceil_mode=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 1024, 1024)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3], strides=[2,2], pads=[1,1,1,1], ceil_mode=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3]
pool_op.strides = [2,2]
pool_op.auto_pad = ""
pool_op.count_include_pad = 0
pool_op.ceil_mode = 1
pool_op.dilations = None
pool_op.pads = [1,1,1,1]
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 10: 2D, asymmetric padding, count_include_pad=1
print("Case 10: 2D, asymmetric padding, count_include_pad=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 1024, 1024)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3], strides=[2,2], pads=[0,1,2,3], count_include_pad=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3]
pool_op.strides = [2,2]
pool_op.auto_pad = ""
pool_op.count_include_pad = 1
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = [0,1,2,3]
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 11: 2D, stride > kernel
print("Case 11: 2D, stride > kernel")
numpy_in = np.random.uniform(-8, 8, (2, 3, 1024, 1024)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[2,2], strides=[3,3], auto_pad="VALID"
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [2,2]
pool_op.strides = [3,3]
pool_op.auto_pad = "VALID"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 12: 3D, auto_pad=SAME_UPPER, count_include_pad=0
print("Case 12: 3D, auto_pad=SAME_UPPER, count_include_pad=0")
numpy_in = np.random.uniform(-8, 8, (2, 3, 64, 64, 64)).astype(np.float32)  # NCDHW
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[2,2,2], strides=[1,1,1], auto_pad="SAME_UPPER"
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [2,2,2]
pool_op.strides = [1,1,1]
pool_op.auto_pad = "SAME_UPPER"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 13: 3D, auto_pad=SAME_LOWER, count_include_pad=1
print("Case 13: 3D, auto_pad=SAME_LOWER, count_include_pad=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 64, 64, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[2,2,2], strides=[1,1,1], auto_pad="SAME_LOWER", count_include_pad=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [2,2,2]
pool_op.strides = [1,1,1]
pool_op.auto_pad = "SAME_LOWER"
pool_op.count_include_pad = 1
pool_op.ceil_mode = 0
pool_op.dilations = None
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 14: 3D, auto_pad=VALID, with dilations
print("Case 14: 3D, auto_pad=VALID, with dilations")
numpy_in = np.random.uniform(-8, 8, (2, 3, 64, 64, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3,3], strides=[2,2,2], auto_pad="VALID", dilations=[2,2,2]
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3,3]
pool_op.strides = [2,2,2]
pool_op.auto_pad = "VALID"
pool_op.count_include_pad = 0
pool_op.ceil_mode = 0
pool_op.dilations = [2,2,2]
pool_op.pads = None
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 15: 3D, explicit padding, ceil_mode=1
print("Case 15: 3D, explicit padding, ceil_mode=1")
numpy_in = np.random.uniform(-8, 8, (2, 3, 64, 64, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1], ceil_mode=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [3,3,3]
pool_op.strides = [2,2,2]
pool_op.auto_pad = ""
pool_op.count_include_pad = 0
pool_op.ceil_mode = 1
pool_op.dilations = None
pool_op.pads = [1,1,1,1,1,1]
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 16: 3D, mixed strides and dilations
print("Case 16: 3D, mixed strides and dilations")
numpy_in = np.random.uniform(-8, 8, (2, 3, 64, 64, 64)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_average_pool(
    numpy_in, kernel_shape=[2,3,2], strides=[1,2,1], auto_pad="",
    pads=[1,0,1,0,1,0], dilations=[1,2,1], ceil_mode=1
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
pool_op.kernel_shape = [2,3,2]
pool_op.strides = [1,2,1]
pool_op.auto_pad = ""
pool_op.count_include_pad = 0
pool_op.ceil_mode = 1
pool_op.dilations = [1,2,1]
pool_op.pads = [1,0,1,0,1,0]
kp_out = pool_op.run(numpy_in)[0]
print(f"{pool_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()
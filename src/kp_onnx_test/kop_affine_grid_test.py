from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_affine_grid import AffineGridOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

affinegrid_op = AffineGridOp(mgr)


def construct_original_grid(data_size, align_corners):
    is_2d = len(data_size) == 2
    size_zeros = np.zeros(data_size)
    original_grid = [np.ones(data_size)]
    for dim, dim_size in enumerate(data_size):
        if align_corners == 1:
            step = 2.0 / (dim_size - 1)
            start = -1
            stop = 1 + 0.0001
            a = np.arange(start, stop, step)
        else:
            step = 2.0 / dim_size
            start = -1 + step / 2
            stop = 1
            a = np.arange(start, stop, step)
        if dim == 0:
            if is_2d:
                y = np.reshape(a, (dim_size, 1)) + size_zeros
                original_grid = [y, *original_grid]
            else:
                z = np.reshape(a, (dim_size, 1, 1)) + size_zeros
                original_grid = [z, *original_grid]
        elif dim == 1:
            if is_2d:
                x = np.reshape(a, (1, dim_size)) + size_zeros
                original_grid = [x, *original_grid]
            else:
                y = np.reshape(a, (1, dim_size, 1)) + size_zeros
                original_grid = [y, *original_grid]
        else:
            x = np.reshape(a, (1, dim_size)) + size_zeros
            original_grid = [x, *original_grid]
    return np.stack(original_grid, axis=2 if is_2d else 3)


def apply_affine_transform(theta_n, original_grid_homo):
    # theta_n: (N, 2, 3) for 2D, (N, 3, 4) for 3D
    # original_grid_homo: (H, W, 3) for 2D, (D, H, W, 4) for 3D
    assert theta_n.ndim == 3, (
        "theta_n shall have shape of (N, 2, 3) for 2D, (N, 3, 4) for 3D"
    )
    if original_grid_homo.ndim == 3:
        N, dim_2d, dim_homo = theta_n.shape
        assert dim_2d == 2 and dim_homo == 3
        H, W, dim_homo = original_grid_homo.shape
        assert dim_homo == 3
        # reshape to [H * W, dim_homo] and then transpose to [dim_homo, H * W]
        original_grid_transposed = np.transpose(
            np.reshape(original_grid_homo, (H * W, dim_homo))
        )
        grid_n = np.matmul(
            theta_n, original_grid_transposed
        )  # shape (N, dim_2d, H * W)
        # transpose to (N, H * W, dim_2d) and then reshape to (N, H, W, dim_2d)
        grid = np.reshape(np.transpose(grid_n, (0, 2, 1)), (N, H, W, dim_2d))
        return grid.astype(np.float32)
    else:
        assert original_grid_homo.ndim == 4
        N, dim_3d, dim_homo = theta_n.shape
        assert dim_3d == 3 and dim_homo == 4
        D, H, W, dim_homo = original_grid_homo.shape
        assert dim_homo == 4
        # reshape to [D * H * W, dim_homo] and then transpose to [dim_homo, D * H * W]
        original_grid_transposed = np.transpose(
            np.reshape(original_grid_homo, (D * H * W, dim_homo))
        )
        grid_n = np.matmul(
            theta_n, original_grid_transposed
        )  # shape (N, dim_3d, D * H * W)
        # transpose to (N, D * H * W, dim_3d) and then reshape to (N, D, H, W, dim_3d)
        grid = np.reshape(np.transpose(grid_n, (0, 2, 1)), (N, D, H, W, dim_3d))
        return grid.astype(np.float32)


def np_affine_grid(theta, size, align_corners=None):
    align_corners = align_corners or 0
    _, _, *data_size = size
    original_grid = construct_original_grid(data_size, align_corners)
    grid = apply_affine_transform(theta, original_grid)
    return grid


# -------- Case 1: data: 2D --------
print("Case 1: data: 2D, align_corners: 0")
theta = np.random.random((1, 2, 3)).astype(np.float32)
size = np.array([1, 3, 3, 4]).astype(np.int64)

start_time = time.time()
np_out = np_affine_grid(theta, size)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
affinegrid_op.align_corners = 0
kp_out = affinegrid_op.run(theta, size)[0]
print(f"{affinegrid_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: data: 2D, align_corners: 1 --------
print("Case 2: data: 2D, align_corners: 1")
theta = np.random.random((8, 2, 3)).astype(np.float32)
size = np.array([1, 3, 8, 12]).astype(np.int64)

start_time = time.time()
np_out = np_affine_grid(theta, size, align_corners=1)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
affinegrid_op.align_corners = 1
kp_out = affinegrid_op.run(theta, size)[0]
print(f"{affinegrid_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: data: 3D, align_corners: 0 --------
print("Case 3: data: 3D, align_corners: 0")
theta = np.random.random((511, 3, 4)).astype(np.float32)
size = np.array([1, 3, 3, 6, 8]).astype(np.int64)

start_time = time.time()
np_out = np_affine_grid(theta, size)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
affinegrid_op.align_corners = 0
kp_out = affinegrid_op.run(theta, size)[0]
print(f"{affinegrid_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: data: 3D, align_corners: 1 --------
print("Case 4: data: 3D, align_corners: 1")
theta = np.random.random((511, 3, 4)).astype(np.float32)
size = np.array([1, 3, 3, 8, 8]).astype(np.int64)

start_time = time.time()
np_out = np_affine_grid(theta, size, align_corners=1)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
affinegrid_op.align_corners = 1
kp_out = affinegrid_op.run(theta, size)[0]
print(f"{affinegrid_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

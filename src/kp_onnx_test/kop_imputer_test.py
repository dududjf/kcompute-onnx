import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_imputer import ImputerOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

imputer_op = ImputerOp(mgr)


def onnx_imputer(x, imputed_value_floats=None, imputed_value_int64s=None, replaced_value_float=None,
                 replaced_value_int64=None):
    if imputed_value_floats is not None and len(imputed_value_floats) > 0:
        values = imputed_value_floats
        replace = replaced_value_float
    elif imputed_value_int64s is not None and len(imputed_value_int64s) > 0:
        values = imputed_value_int64s
        replace = replaced_value_int64
    else:
        raise ValueError("Missing are not defined.")

    if isinstance(values, list):
        values = np.array(values)
    if len(x.shape) != 2:
        raise TypeError(f"x must be a matrix but shape is {x.shape}")
    if values.shape[0] not in (x.shape[1], 1):
        raise TypeError(  # pragma: no cover
            f"Dimension mismatch {values.shape[0]} != {x.shape[1]}"
        )
    x = x.copy()
    if np.isnan(replace):
        for i in range(x.shape[1]):
            val = values[min(i, values.shape[0] - 1)]
            x[np.isnan(x[:, i]), i] = val
    else:
        for i in range(x.shape[1]):
            val = values[min(i, values.shape[0] - 1)]
            x[x[:, i] == replace, i] = val
    return x


# -------- Case 1: --------
print("Case 1:")
x = np.array([[1.0, 2.0, 3.0, 4.0],
              [5.0, 1.0, 7.0, 1.0],
              [9.0, 10.0, 1.0, 12.0]], dtype=np.float32)
imputed_value_floats = [2.0, 3.0, 4.0, 8.0]
replaced_value_float = 1.0

start_time = time.time()
np_out = onnx_imputer(x, imputed_value_floats=imputed_value_floats, replaced_value_float=replaced_value_float)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
imputer_op.set_imputed_values(imputed_value_floats=imputed_value_floats, replaced_value_float=replaced_value_float)
kp_out = imputer_op.run(x)[0]
print(f"{imputer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: --------
print("Case 2: Replace NaN with specific values per column")
x = np.array([[1.0, np.nan, 3.0, np.nan],
              [5.0, 6.0, np.nan, 8.0],
              [np.nan, 10.0, 11.0, 12.0]], dtype=np.float32)
imputed_value_floats = [1.0, 3.0, 10.0, 12.0]
replaced_value_float = np.nan

start_time = time.time()
np_out = onnx_imputer(x, imputed_value_floats=imputed_value_floats, replaced_value_float=replaced_value_float)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
imputer_op.set_imputed_values(imputed_value_floats=imputed_value_floats, replaced_value_float=replaced_value_float)
kp_out = imputer_op.run(x)[0]
print(f"{imputer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: Replace specific value (not NaN) --------
print("Case 3: Replace specific value (-1)")
x = np.array([[-1.0, 2.0, -1.0, 4.0],
              [-1.0, 6.0, 7.0, -1.0],
              [9.0, -1.0, 11.0, 12.0]], dtype=np.float32)
imputed_value_floats = [10.0]
replaced_value_float = -1.0

start_time = time.time()
np_out = onnx_imputer(x, imputed_value_floats=imputed_value_floats, replaced_value_float=replaced_value_float)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
imputer_op.set_imputed_values(imputed_value_floats=imputed_value_floats, replaced_value_float=replaced_value_float)
kp_out = imputer_op.run(x)[0]
print(f"{imputer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: Replace specific value with column-specific values --------
print("Case 4: Replace -999 with mean-like values")
x = np.array([[1, 2, 3, 4],
              [-999, 6, -999, 8],
              [9, -999, 11, -999]]).astype(np.int64)

# Simulate mean imputation (pre-calculated means)
imputed_value_int64s = [10]
replaced_value_int64 = -999

start_time = time.time()
np_out = onnx_imputer(x, imputed_value_int64s=imputed_value_int64s, replaced_value_int64=replaced_value_int64)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
imputer_op.set_imputed_values(imputed_value_floats=[], imputed_value_int64s=imputed_value_int64s, replaced_value_int64=replaced_value_int64)
kp_out = imputer_op.run(x)[0]
print(f"{imputer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: Replace specific value with column-specific values --------
print("Case 5: Replace -999 with mean-like values")
x = np.array([[4, 5, 6, 7],
              [-999, 6, -999, 8],
              [9, -999, 11, -999]]).astype(np.int64)

# Simulate mean imputation (pre-calculated means)
imputed_value_int64s = [10, 3, 5, 10]
replaced_value_int64 = -999

start_time = time.time()
np_out = onnx_imputer(x, imputed_value_int64s=imputed_value_int64s, replaced_value_int64=replaced_value_int64)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
imputer_op.set_imputed_values(imputed_value_int64s=imputed_value_int64s, replaced_value_int64=replaced_value_int64)
kp_out = imputer_op.run(x)[0]
print(f"{imputer_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
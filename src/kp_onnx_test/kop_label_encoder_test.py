import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_label_encoder import LabelEncoderOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())


def np_label_encoder(
        x,
        default_float=None,
        default_int64=None,
        default_string=None,
        default_tensor=None,
        keys_floats=None,
        keys_int64s=None,
        keys_strings=None,
        values_floats=None,
        values_int64s=None,
        values_strings=None,
        keys_tensor=None,
        values_tensor=None,
):
    keys = keys_floats or keys_int64s or keys_strings or keys_tensor
    values = values_floats or values_int64s or values_strings or values_tensor
    classes = dict(zip(keys, values))

    if values is values_tensor:
        defval = default_tensor.item()
        otype = default_tensor.dtype
    elif values is values_floats:
        defval = default_float
        otype = np.float32
    elif values is values_int64s:
        defval = default_int64
        otype = np.int64
    elif values is values_strings:
        defval = default_string
        otype = np.str_
        if not isinstance(defval, str):
            defval = ""
    lookup_func = np.vectorize(lambda x: classes.get(x, defval), otypes=[otype])
    output = lookup_func(x)
    if output.dtype == object:
        output = output.astype(np.str_)
    return output


# -------- Case 1: Float keys to float values --------
print("Case 1: Float keys to float values")
x = np.array([1.0, 2.0, 3.0, 1.0, 5.0, 2.0], dtype=np.float32)
keys_floats = [1.0, 2.0, 3.0, 4.0]
values_floats = [10.0, 20.0, 30.0, 40.0]
default_float = -1.0

start_time = time.time()
np_out = np_label_encoder(x, default_float=default_float, keys_floats=keys_floats, values_floats=values_floats)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
labelencoder_op = LabelEncoderOp(mgr, default_float=default_float, keys_floats=keys_floats, values_floats=values_floats)
kp_out = labelencoder_op.run(x)[0]
print(f"{labelencoder_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: With default value --------
print("Case 2: Using default value for unknown keys")
x = np.array([[1.0, 2.0, 99.0, 1.0, np.nan, 2.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=np.float32)
keys_floats = [1.0, 2.0, 2.0]
values_floats = [100.0, 200.0, 300.0]
default_float = -0.0

start_time = time.time()
np_out = np_label_encoder(x, default_float=default_float, keys_floats=keys_floats, values_floats=values_floats)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
labelencoder_op.set_keys_values(default_float=default_float, keys_floats=keys_floats, values_floats=values_floats)
kp_out = labelencoder_op.run(x)[0]
print(f"{labelencoder_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: Int64 keys to int64 values --------
print("Case 3: Int64 keys to int64 values")
x = np.array([[0, 1, 2, 0, 3, 1], [2, 4, 10, 6, 7, 3]], dtype=np.int64)
keys_int64s = [0, 1, 2, 3]
values_int64s = [100, 200, 300, 400]
default_int64 = -999

start_time = time.time()
np_out = np_label_encoder(x, default_int64=default_int64, keys_int64s=keys_int64s, values_int64s=values_int64s)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
labelencoder_op.set_keys_values(default_int64=default_int64, keys_int64s=keys_int64s, values_int64s=values_int64s)
kp_out = labelencoder_op.run(x.astype(np.float32))[0]
print(f"{labelencoder_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: 1D, float key --------
print("Case 4: 1D tensor, float key")
x = np.array([1.0, 2.0, 3.0, 1.0, 5.0, 2.0], dtype=np.float32)
keys_floats = [1.0, 2.0, 3.0]
values_floats = [10.0, 20.0, 30.0]
default_float = 0.0

start_time = time.time()
np_out = np_label_encoder(x, default_float=default_float, keys_floats=keys_floats, values_floats=values_floats)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
labelencoder_op.set_keys_values(default_float=default_float, keys_floats=keys_floats, values_floats=values_floats)
kp_out = labelencoder_op.run(x)[0]
print(f"{labelencoder_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: 1D, int64 key --------
print("Case 5: 1D tensor, int64 key")
x = np.array([1, 2, 3, 1, 5, 2], dtype=np.int64)
keys_int64s = [1, 2, 3]
values_int64s = [10, 20, 30]
default_int64 = 0

start_time = time.time()
np_out = np_label_encoder(x, default_int64=default_int64, keys_int64s=keys_int64s, values_int64s=values_int64s)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
labelencoder_op.set_keys_values(default_int64=default_int64, keys_int64s=keys_int64s, values_int64s=values_int64s)
kp_out = labelencoder_op.run(x)[0]
print(f"{labelencoder_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6: tensor key --------
print("Case 6: tensor key")
x = np.array([[99.0, 88.0, 77.0], [99.0, 88.0, 77.0]], dtype=np.float32)
keys_tensor = np.array([77.0, 2.0, 99.0], dtype=np.float32)
values_tensor = np.array([10.0, 20.0, 30.0], dtype=np.float32)
default_tensor = np.array([-0.0], dtype=np.float32)

start_time = time.time()
np_out = np_label_encoder(x, default_tensor=default_tensor, keys_tensor=keys_tensor, values_tensor=values_tensor)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
labelencoder_op.set_keys_values(default_tensor=default_tensor, keys_tensor=keys_tensor, values_tensor=values_tensor)
kp_out = labelencoder_op.run(x)[0]
print(f"{labelencoder_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

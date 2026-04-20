from kp import Manager
import numpy as np
from kp_onnx_ssbo.kop_optional_get_element import OptionalGetElementOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

opt_get_op = OptionalGetElementOp(mgr)

print("\nCase 1: Basic test - get element from optional input")
numpy_in = np.array([1, 2, 3, 4], dtype=np.float32)
print("Input shape:", numpy_in.shape)
print("Input:", numpy_in)

result = opt_get_op.run(numpy_in)
print("Output shape:", result[0].shape)
print("Output:", result[0])
print("Match:", np.allclose(numpy_in, result[0]))
print("Op repr:", opt_get_op)
print()

print("Case 2: Test assertion with empty inputs (should fail)")
try:
    result = opt_get_op.run()
    print("ERROR: Should have raised assertion!")
except AssertionError as e:
    print(f"Correctly caught assertion: {e}")


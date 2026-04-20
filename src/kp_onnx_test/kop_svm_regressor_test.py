from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_svm_regressor import SVMRegressorOp


# --------------------------
# ONNX SVM Regressor baseline (self-contained, copied from op_svm_regressor.py and op_svm_helper.py)
# --------------------------
class _SVMAttributes:
    def __init__(self):
        pass

    def add(self, name, value):
        if isinstance(value, list) and name not in {"kernel_params"}:
            value = np.array(value, dtype=np.float32)
        setattr(self, name, value)


class _SVMCommon:
    def __init__(self, **kwargs):
        self.atts = _SVMAttributes()
        for name, value in kwargs.items():
            self.atts.add(name, value)

        if getattr(self.atts, "kernel_params", None) is not None:
            self.gamma_ = self.atts.kernel_params[0]
            self.coef0_ = self.atts.kernel_params[1]
            self.degree_ = int(self.atts.kernel_params[2])
        else:
            self.gamma_ = 0.0
            self.coef0_ = 0.0
            self.degree_ = 0

    def kernel_dot(self, pA: np.ndarray, pB: np.ndarray, kernel: str) -> np.ndarray:
        k = kernel.lower()
        if k == "poly":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return s ** self.degree_
        if k == "sigmoid":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return np.tanh(s)
        if k == "rbf":
            diff = pA - pB
            s = (diff * diff).sum()
            return np.exp(-self.gamma_ * s)
        if k == "linear":
            return np.dot(pA, pB)
        raise ValueError(f"Unexpected kernel={kernel!r}.")

    def run_reg(self, X: np.ndarray) -> np.ndarray:
        z = np.empty((X.shape[0], 1), dtype=X.dtype)

        if self.atts.n_supports > 0:
            # Normal mode: use support vectors and kernel
            kernel_type_ = self.atts.kernel_type
            sv = self.atts.support_vectors.reshape((self.atts.n_supports, -1))
            for n in range(X.shape[0]):
                s = 0.0
                for j in range(self.atts.n_supports):
                    d = self.kernel_dot(X[n], sv[j], kernel_type_)
                    s += self.atts.coefficients[j] * d
                s += self.atts.rho[0]
                z[n, 0] = s
        else:
            # SVM_LINEAR mode: use coefficients as weight vector
            # Formula: output = X · coefficients + rho[0]
            kernel_type_ = "LINEAR"
            for n in range(X.shape[0]):
                s = self.kernel_dot(X[n], self.atts.coefficients, kernel_type_)
                s += self.atts.rho[0]
                z[n, 0] = s

        # Handle one_class if needed (not used in regressor, but kept for compatibility)
        if getattr(self.atts, "one_class", False):
            z = np.where(z > 0, 1, -1)

        return z


def onnx_reference_svm_regressor(
    X,
    coefficients=None,
    kernel_params=None,
    kernel_type="LINEAR",
    n_targets=None,
    n_supports=None,
    one_class=None,
    post_transform=None,
    rho=None,
    support_vectors=None,
):
    X = np.array(X).astype(np.float32)
    # Flatten X from second dimension onwards
    X = X.reshape((X.shape[0], -1))

    svm = _SVMCommon(
        coefficients=coefficients,
        kernel_params=kernel_params,
        kernel_type=kernel_type,
        n_targets=n_targets,
        n_supports=n_supports if n_supports is not None else 0,
        one_class=one_class if one_class is not None else 0,
        post_transform=post_transform,
        rho=rho,
        support_vectors=support_vectors,
    )

    res = svm.run_reg(X)

    if post_transform in (None, "NONE"):
        return res
    raise NotImplementedError(f"post_transform={post_transform!r} not implemented.")



np.random.seed(42)
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

svm_op = SVMRegressorOp(mgr)

# Case 1: SVM_LINEAR mode (n_supports=0) — should compute X·coefficients+rho
print("Case 1: SVM_LINEAR mode (n_supports=0) — formula: X·coefficients+rho")
numpy_in = np.random.uniform(-1, 1, (10000, 6400)).astype(np.float32)
# When n_supports=0, kernel_type is ignored, always uses LINEAR
# Output = X · coefficients + rho[0]
coefs = np.random.uniform(-1, 1, (6400,)).astype(np.float32)
rho = [3.14159]

start_time = time.time()
numpy_out = onnx_reference_svm_regressor(
    numpy_in,
    coefficients=coefs,
    kernel_params=[0.5, 0, 0],  # ignored when n_supports=0
    kernel_type="RBF",  # ignored when n_supports=0
    n_supports=0,
    post_transform="NONE",
    rho=rho,
    support_vectors=None,
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
print("NumPy output range: [%.4f, %.4f]" % (numpy_out.min(), numpy_out.max()))

start_time = time.time()
svm_op.coefficients = coefs
svm_op.kernel_params = [0.5, 0, 0]
svm_op.kernel_type = "RBF"  # ignored when n_supports=0
svm_op.n_supports = 0
svm_op.one_class = 0
svm_op.post_transform = "NONE"
svm_op.rho = rho
svm_op.support_vectors = None
kp_out = svm_op.run(numpy_in)[0]
print(f"{svm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Kompute output range: [%.4f, %.4f]" % (kp_out.min(), kp_out.max()))
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: SVC mode with LINEAR kernel
print("Case 2: SVC mode with LINEAR kernel")
numpy_in = np.random.uniform(-1, 1, (2048, 512)).astype(np.float32)
n_supports = 256
support_vectors = np.random.uniform(-1, 1, (n_supports, 512)).astype(np.float32)
coefficients = np.random.uniform(-1, 1, (n_supports,)).astype(np.float32)
rho = [0.15]

start_time = time.time()
numpy_out = onnx_reference_svm_regressor(
    numpy_in,
    coefficients=coefficients,
    kernel_params=None,
    kernel_type="LINEAR",
    n_supports=n_supports,
    post_transform="NONE",
    rho=rho,
    support_vectors=support_vectors,
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.coefficients = coefficients
svm_op.kernel_params = None
svm_op.kernel_type = "LINEAR"
svm_op.n_supports = n_supports
svm_op.rho = rho
svm_op.support_vectors = support_vectors
kp_out = svm_op.run(numpy_in)[0]
print(f"{svm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: SVC mode with POLY kernel (degree=3)
print("Case 3: SVC mode with POLY kernel (degree=3)")
numpy_in = np.random.uniform(-0.5, 0.5, (4096, 1024)).astype(np.float32)
n_supports = 128
support_vectors = np.random.uniform(-0.5, 0.5, (n_supports, 1024)).astype(np.float32)
coefficients = np.random.uniform(-1, 1, (n_supports,)).astype(np.float32)
kernel_params = [0.2, 1.0, 3]  # gamma, coef0, degree
rho = [0.05]

start_time = time.time()
numpy_out = onnx_reference_svm_regressor(
    numpy_in,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="POLY",
    n_supports=n_supports,
    post_transform="NONE",
    rho=rho,
    support_vectors=support_vectors,
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "POLY"
svm_op.n_supports = n_supports
svm_op.rho = rho
svm_op.support_vectors = support_vectors
kp_out = svm_op.run(numpy_in)[0]
print(f"{svm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: SVC mode with RBF kernel
print("Case 4: SVC mode with RBF kernel")
numpy_in = np.random.uniform(-1, 1, (8092, 2048)).astype(np.float32)
n_supports = 256
support_vectors = np.random.uniform(-1, 1, (n_supports, 2048)).astype(np.float32)
coefficients = np.random.uniform(-1, 1, (n_supports,)).astype(np.float32)
kernel_params = [0.5, 0.0, 0]  # gamma, coef0, degree
rho = [-0.1]

start_time = time.time()
numpy_out = onnx_reference_svm_regressor(
    numpy_in,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    n_supports=n_supports,
    post_transform="NONE",
    rho=rho,
    support_vectors=support_vectors,
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "RBF"
svm_op.n_supports = n_supports
svm_op.rho = rho
svm_op.support_vectors = support_vectors
kp_out = svm_op.run(numpy_in)[0]
print(f"{svm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: SVC mode with SIGMOID kernel
print("Case 5: SVC mode with SIGMOID kernel")
numpy_in = np.random.uniform(-1, 1, (8092, 2048)).astype(np.float32)
n_supports = 128
support_vectors = np.random.uniform(-1, 1, (n_supports, 2048)).astype(np.float32)
coefficients = np.random.uniform(-1, 1, (n_supports,)).astype(np.float32)
kernel_params = [0.4, -0.3, 0]  # gamma, coef0, degree
rho = [0.08]

start_time = time.time()
numpy_out = onnx_reference_svm_regressor(
    numpy_in,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="SIGMOID",
    n_supports=n_supports,
    post_transform="NONE",
    rho=rho,
    support_vectors=support_vectors,
)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "SIGMOID"
svm_op.n_supports = n_supports
svm_op.rho = rho
svm_op.support_vectors = support_vectors
kp_out = svm_op.run(numpy_in)[0]
print(f"{svm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: SVC mode with unsupported kernel type — expects ValueError (line 278-280)
print("Case 6: SVC mode with unsupported kernel type (expects ValueError)")
numpy_in_6 = np.random.uniform(-1, 1, (16, 8)).astype(np.float32)
svm_op.coefficients = np.random.uniform(-1, 1, (4,)).astype(np.float32)
svm_op.kernel_params = [0.5, 0.0, 0]
svm_op.kernel_type = "UNKNOWN_KERNEL"
svm_op.n_supports = 4
svm_op.rho = [0.0]
svm_op.support_vectors = np.random.uniform(-1, 1, (4, 8)).astype(np.float32)
try:
    svm_op.run(numpy_in_6)
    print("ERROR: expected ValueError was not raised!")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()


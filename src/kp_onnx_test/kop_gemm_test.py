import kp
import numpy as np
import time
from kp_onnx_ssbo.kop_gemm import GemmOp


def onnx_reference_gemm(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0):
    """Reference implementation from ONNX op_gemm.py"""
    def _gemm00(a, b, c, alpha, beta):
        o = np.dot(a, b) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    def _gemm01(a, b, c, alpha, beta):
        o = np.dot(a, b.T) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    def _gemm10(a, b, c, alpha, beta):
        o = np.dot(a.T, b) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    def _gemm11(a, b, c, alpha, beta):
        o = np.dot(a.T, b.T) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    if transA:
        _meth = _gemm11 if transB else _gemm10
    else:
        _meth = _gemm01 if transB else _gemm00

    return _meth(a, b, c, alpha, beta).astype(a.dtype)


device_id = 0
mgr = kp.Manager(device_id)
print(mgr.get_device_properties())

# Case 1: transA=0, transB=0, C=None
print("Case 1: transA=0, transB=0, C=None")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=None, alpha=1.0, beta=1.0, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: transA=0, transB=0, alpha!=1
print("Case 2: transA=0, transB=0, alpha!=1")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=None, alpha=2.5, beta=1.0, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=2.5, beta=1.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: transA=0, transB=0, C!=None, beta!=0
print("Case 3: transA=0, transB=0, C!=None, beta!=0")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(256, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.0, beta=1.5, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.5, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: transA=0, transB=0, alpha!=1, beta!=1, C!=None
print("Case 4: transA=0, transB=0, alpha!=1, beta!=1, C!=None")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(256, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=0.8, beta=0.3, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=0.8, beta=0.3, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: transA=1, transB=0
print("Case 5: transA=1, transB=0")
A = np.random.randn(512, 256).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=None, alpha=1.0, beta=1.0, transA=1, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=1, transB=0)
kp_out = gemm_op.run(A, B)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: transA=0, transB=1
print("Case 6: transA=0, transB=1")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(384, 512).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=None, alpha=1.0, beta=1.0, transA=0, transB=1)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=0, transB=1)
kp_out = gemm_op.run(A, B)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: transA=1, transB=1
print("Case 7: transA=1, transB=1")
A = np.random.randn(512, 256).astype(np.float32)
B = np.random.randn(384, 512).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=None, alpha=1.0, beta=1.0, transA=1, transB=1)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=1, transB=1)
kp_out = gemm_op.run(A, B)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 8: transA=1, transB=0, alpha!=1, beta!=1, C!=None
print("Case 8: transA=1, transB=0, alpha!=1, beta!=1, C!=None")
A = np.random.randn(512, 256).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(256, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=2.0, beta=0.5, transA=1, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=2.0, beta=0.5, transA=1, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 9: transA=0, transB=1, alpha!=1, beta!=1, C!=None
print("Case 9: transA=0, transB=1, alpha!=1, beta!=1, C!=None")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(384, 512).astype(np.float32)
C = np.random.randn(256, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.5, beta=0.8, transA=0, transB=1)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.5, beta=0.8, transA=0, transB=1)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 10: transA=1, transB=1, alpha!=1, beta!=1, C!=None
print("Case 10: transA=1, transB=1, alpha!=1, beta!=1, C!=None")
A = np.random.randn(512, 256).astype(np.float32)
B = np.random.randn(384, 512).astype(np.float32)
C = np.random.randn(256, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=0.7, beta=0.4, transA=1, transB=1)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=0.7, beta=0.4, transA=1, transB=1)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 11: 按行广播
print("Case 11: broadcast C (row vector)")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(1, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.0, beta=1.0, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 12: 按列广播
print("Case 12: broadcast C (column vector)")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(256, 1).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.0, beta=1.0, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 13: beta=0 (C ignored)
print("Case 13: beta=0 (C ignored)")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(256, 384).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.0, beta=0.0, transA=0, transB=0)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
gemm_op = GemmOp(mgr, alpha=1.0, beta=0.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 14: C 标量广播
print("Case 14: C is scalar (0D)")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.array(2.5, dtype=np.float32)  # 0D scalar
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.0, beta=1.0, transA=0, transB=0)
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 15: C 1D 行向量广播
print("Case 15: C is 1D row vector (N,)")
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 384).astype(np.float32)
C = np.random.randn(384).astype(np.float32)  # shape (384,), not (1,384)
numpy_out = onnx_reference_gemm(A, B, c=C, alpha=1.0, beta=1.0, transA=0, transB=0)
gemm_op = GemmOp(mgr, alpha=1.0, beta=1.0, transA=0, transB=0)
kp_out = gemm_op.run(A, B, C)[0]
print(f"{gemm_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()
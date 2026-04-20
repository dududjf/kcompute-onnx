from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_dft import DFTOp


def onnx_reference_dft(x, dft_length=None, axis=-2, inverse=False, onesided=False):
    """参考实现：基于ONNX规范的DFT"""
    # Convert to positive axis
    axis = axis % len(x.shape)
    if dft_length is None:
        dft_length = x.shape[axis]

    if inverse:
        result = _cifft(x, dft_length, axis=axis, onesided=onesided)
    else:
        result = _cfft(x, dft_length, axis=axis, onesided=onesided, normalize=False)

    return result.astype(x.dtype)


def _fft(x: np.ndarray, fft_length: int, axis: int) -> np.ndarray:
    """Compute the FFT return the real representation of the complex result."""
    transformed = np.fft.fft(x, n=fft_length, axis=axis)
    real_frequencies = np.real(transformed)
    imaginary_frequencies = np.imag(transformed)
    return np.concatenate(
        (real_frequencies[..., np.newaxis], imaginary_frequencies[..., np.newaxis]),
        axis=-1,
    )


def _cfft(x: np.ndarray, fft_length: int, axis: int, onesided: bool, normalize: bool) -> np.ndarray:
    if x.shape[-1] == 1:
        # The input contains only the real part
        signal = x
    else:
        # The input is a real representation of a complex signal
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        signal = real + 1j * imag

    complex_signals = np.squeeze(signal, -1)
    result = _fft(complex_signals, fft_length, axis=axis)
    # Post process the result based on arguments
    if onesided:
        slices = [slice(0, a) for a in result.shape]
        slices[axis] = slice(0, result.shape[axis] // 2 + 1)
        result = result[tuple(slices)]
    if normalize:
        result /= fft_length
    return result


def _ifft(x: np.ndarray, fft_length: int, axis: int, onesided: bool) -> np.ndarray:
    signals = np.fft.ifft(x, fft_length, axis=axis)
    real_signals = np.real(signals)
    imaginary_signals = np.imag(signals)
    merged = np.concatenate(
        (real_signals[..., np.newaxis], imaginary_signals[..., np.newaxis]),
        axis=-1,
    )
    if onesided:
        slices = [slice(a) for a in merged.shape]
        slices[axis] = slice(0, merged.shape[axis] // 2 + 1)
        return merged[tuple(slices)]
    return merged


def _cifft(x: np.ndarray, fft_length: int, axis: int, onesided: bool = False) -> np.ndarray:
    if x.shape[-1] == 1:
        frequencies = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        frequencies = real + 1j * imag
    complex_frequencies = np.squeeze(frequencies, -1)
    return _ifft(complex_frequencies, fft_length, axis=axis, onesided=onesided)


# 测试执行部分
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

def run_dft(dft_op, input_data, dft_length=None, axis=-2, inverse=0, onesided=0):
    """Helper function to run DFT with axis as input"""
    dft_op.inverse = inverse
    dft_op.onesided = onesided

    inputs = [input_data]
    if dft_length is not None:
        inputs.append(np.array([dft_length], dtype=np.int64))
    else:
        inputs.append(np.array([], dtype=np.int64))  # Empty tensor for no dft_length
    inputs.append(np.array([axis], dtype=np.int64))

    return dft_op.run(*inputs)[0]

dft_op = DFTOp(mgr)

# Case 1: 1D DFT, real input, default axis
print("Case 1: 1D DFT, real input, axis=-2")
numpy_in = np.random.uniform(-1, 1, (16, 256, 1)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=-2, inverse=False, onesided=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=-2, inverse=0, onesided=0)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=3e-3, atol=3e-3))
print()

# Case 2: 1D DFT, complex input, default axis
print("Case 2: 1D DFT, complex input, axis=-2")
numpy_in = np.random.uniform(-1, 1, (16, 256, 2)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=-2, inverse=False, onesided=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=-2, inverse=0, onesided=0)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=3e-3, atol=3e-3))
print()

# Case 3: 1D IDFT, complex input
print("Case 3: 1D IDFT, complex input, axis=-2")
numpy_in = np.random.uniform(-1, 1, (16, 256, 2)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=-2, inverse=True, onesided=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=-2, inverse=1, onesided=0)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 1D DFT, onesided output
print("Case 4: 1D DFT, onesided=True, axis=-2")
numpy_in = np.random.uniform(-1, 1, (16, 256, 2)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=-2, inverse=False, onesided=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=-2, inverse=0, onesided=1)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=3e-3, atol=3e-3))
print()

# Case 5: 2D DFT, complex input, axis=0
print("Case 5: 2D DFT, complex input, axis=0")
numpy_in = np.random.uniform(-1, 1, (128, 128, 2)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=0, inverse=False, onesided=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=0, inverse=0, onesided=0)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-3, atol=1e-3))
print()

# Case 6: 2D DFT, complex input, axis=1
print("Case 6: 2D DFT, complex input, axis=1")
numpy_in = np.random.uniform(-1, 1, (128, 128, 2)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=1, inverse=False, onesided=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=1, inverse=0, onesided=0)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-3, atol=1e-3))
print()

# Case 7: Real input with onesided
print("Case 7: Real input with onesided=True")
numpy_in = np.random.uniform(-1, 1, (16, 256, 1)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, axis=-2, inverse=False, onesided=True)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, axis=-2, inverse=0, onesided=1)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=3e-3, atol=3e-3))
print()

# Case 8: DFT-IDFT round trip
print("Case 8: DFT-IDFT round trip test")
numpy_in = np.random.uniform(-1, 1, (16, 128, 2)).astype(np.float32)

# Forward DFT
dft_result = run_dft(dft_op, numpy_in, axis=-2, inverse=0, onesided=0)

# Inverse DFT
idft_result = run_dft(dft_op, dft_result, axis=-2, inverse=1, onesided=0)

print("Input shape:", numpy_in.shape)
print("DFT result shape:", dft_result.shape)
print("IDFT result shape:", idft_result.shape)
print("Max error (input vs IDFT):", np.abs(numpy_in - idft_result).max())
print("All close (input vs IDFT):", np.allclose(numpy_in, idft_result, rtol=1e-4, atol=1e-4))
print()

# Case 9: 显式传入 dft_length < input_n
# dft_length=128 < input_n=256，只取前 128 个样本参与计算
print("Case 9: 显式传入 dft_length < input_n")
numpy_in = np.random.uniform(-1, 1, (16, 256, 2)).astype(np.float32)
dft_length = 128  # 截断至 128 点
start_time = time.time()
numpy_out = onnx_reference_dft(numpy_in, dft_length=dft_length, axis=-2, inverse=False, onesided=False)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
start_time = time.time()
kp_out = run_dft(dft_op, numpy_in, dft_length=dft_length, axis=-2, inverse=0, onesided=0)
print(f"{dft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=3e-3, atol=3e-3))
print()


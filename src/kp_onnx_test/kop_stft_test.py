import kp
import numpy as np
import time
from kp_onnx.kop_stft import STFTOp


def _concat(*args, axis=0):  # type: ignore
    return np.concatenate(args, axis=axis)


def _unsqueeze(a, axis):  # type: ignore
    try:
        return np.expand_dims(a, axis=axis)
    except TypeError:
        if len(axis) == 1:
            return np.expand_dims(a, axis=tuple(axis)[0])
        for x in reversed(axis):
            a = np.expand_dims(a, axis=x)
        return a


def _slice(
        data: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        axes: np.ndarray | None = None,
        steps: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(starts, list):
        starts = np.array(starts)
    if isinstance(ends, list):
        ends = np.array(ends)
    if isinstance(axes, list):
        axes = np.array(axes)
    if isinstance(steps, list):
        steps = np.array(steps)
    if len(starts.shape) == 0:
        starts = np.array([starts])
    if len(ends.shape) == 0:
        ends = np.array([ends])
    if axes is None:
        if steps is None:
            slices = [slice(s, e) for s, e in zip(starts, ends)]
        else:
            slices = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]
    else:
        if steps is None:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a in zip(starts, ends, axes):
                slices[a] = slice(s, e)
        else:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a, d in zip(starts, ends, axes, steps):
                slices[a] = slice(s, e, d)
    return data[tuple(slices)]


def _concat_from_sequence(seq: list, axis: int, new_axis: int = 0) -> np.ndarray:
    if new_axis == 1:
        if axis == -1:
            seq2 = [s[..., np.newaxis] for s in seq]
            res = np.concatenate(seq2, axis=-1)
        else:
            seq2 = [np.expand_dims(s, axis) for s in seq]
            res = np.concatenate(seq2, axis=axis)
    else:
        res = np.concatenate(seq, axis=axis)
    return res


def _fft(x: np.ndarray, fft_length: int, axis: int) -> np.ndarray:
    transformed = np.fft.fft(x, n=fft_length, axis=axis)
    real_frequencies = np.real(transformed)
    imaginary_frequencies = np.imag(transformed)
    return np.concatenate(
        (real_frequencies[..., np.newaxis], imaginary_frequencies[..., np.newaxis]),
        axis=-1,
    )


def _dft(
        x: np.ndarray,
        fft_length: int,
        axis: int,
        onesided: bool,
        normalize: bool,
) -> np.ndarray:
    if x.shape[-1] == 1:
        signal = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        signal = real + 1j * imag

    complex_signals = np.squeeze(signal, -1)
    result = _fft(complex_signals, fft_length, axis=axis)
    if onesided:
        slices = [slice(0, a) for a in result.shape]
        slices[axis] = slice(0, result.shape[axis] // 2 + 1)
        result = result[tuple(slices)]
    if normalize:
        result /= fft_length
    return result


def _stft(x, fft_length: int, hop_length, n_frames, window, onesided=False):
    last_axis = len(x.shape) - 1
    axis = [-2]
    axis2 = [-3]
    window_size = window.shape[0]

    seq = []
    for fs in range(n_frames):
        begin = fs * hop_length
        end = begin + window_size
        sliced_x = _slice(x, np.array([begin]), np.array([end]), axis)

        new_dim = sliced_x.shape[-2:-1]
        missing = (window_size - new_dim[0],)
        new_shape = sliced_x.shape[:-2] + missing + sliced_x.shape[-1:]
        cst = np.zeros(new_shape, dtype=x.dtype)
        pad_sliced_x = _concat(sliced_x, cst, axis=-2)

        un_sliced_x = _unsqueeze(pad_sliced_x, axis2)
        seq.append(un_sliced_x)

    new_x = _concat_from_sequence(seq, axis=-3, new_axis=0)

    shape_x = new_x.shape
    shape_x_short = shape_x[:-2]
    shape_x_short_one = tuple(1 for _ in shape_x_short)
    window_shape = (*shape_x_short_one, window_size, 1)
    weights = np.reshape(window, window_shape)
    weighted_new_x = new_x * weights

    result = _dft(
        weighted_new_x, fft_length, last_axis, onesided=onesided, normalize=False
    )

    return result


def np_stft(x, frame_step, window=None, frame_length=None, onesided=True):
    if x.ndim == 1:
        x_in = x[:, None]
    elif x.ndim == 2:
        x_in = x[..., None]
    else:
        x_in = x

    if frame_length is None:
        if window is None:
            frame_length = x.shape[-1]
        else:
            frame_length = window.shape[0]

    hop_length = int(frame_step)

    if window is None:
        window = np.ones((frame_length,), dtype=x.dtype)

    n_frames = 1 + (x.shape[-1] - frame_length) // hop_length

    res = _stft(x_in, int(frame_length), hop_length, int(n_frames), window, onesided=onesided)

    return res.astype(x.dtype)

device_id = 0
mgr = kp.Manager(device_id)
print(mgr.get_device_properties())
stft_op = STFTOp(mgr, onesided=1)


print('Case 1: Batched signal, No Window (ones), Default length')
numpy_in = np.random.random((2, 128)).astype(np.float32)
step_in = np.array([32], dtype=np.int64)

start_time = time.time()
np_out = np_stft(numpy_in, step_in[0], window=None, frame_length=None, onesided=True)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = stft_op.run(numpy_in, step_in)[0]
print(f"{stft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 2: With Window, Inferred Frame Length')
numpy_in = np.random.random((3, 100)).astype(np.float32)
step_in = np.array([10], dtype=np.int64)
window_in = np.random.random((16,)).astype(np.float32)

start_time = time.time()
np_out = np_stft(numpy_in, step_in[0], window=window_in, frame_length=None, onesided=True)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = stft_op.run(numpy_in, step_in, window_in)[0]
print(f"{stft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 3: Explicit Frame Length (Matched to Window)')
numpy_in = np.random.random((1, 256)).astype(np.float32)
step_in = np.array([64], dtype=np.int64)
window_in = np.random.random((32,)).astype(np.float32)
len_in = np.array([32], dtype=np.int64)

start_time = time.time()
np_out = np_stft(numpy_in, step_in[0], window=window_in, frame_length=len_in[0], onesided=True)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = stft_op.run(numpy_in, step_in, window_in, len_in)[0]
print(f"{stft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 4: Two-Sided Output')
stft_op_twosided = STFTOp(mgr, onesided=0)
numpy_in = np.random.random((2, 60)).astype(np.float32)
step_in = np.array([15], dtype=np.int64)
window_in = np.random.random((20,)).astype(np.float32)

start_time = time.time()
np_out = np_stft(numpy_in, step_in[0], window=window_in, frame_length=None, onesided=False)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = stft_op_twosided.run(numpy_in, step_in, window_in)[0]
print(f"{stft_op_twosided}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 5: 1D Input Signal')
numpy_in = np.random.random((128,)).astype(np.float32)
step_in = np.array([32], dtype=np.int64)
window_in = np.random.random((32,)).astype(np.float32)

start_time = time.time()
np_out = np_stft(numpy_in, step_in[0], window=window_in, frame_length=None, onesided=True)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = stft_op.run(numpy_in, step_in, window_in)[0]
print(f"{stft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))


print('Case 6: frame_length > window_length (broadcast window)')
numpy_in = np.random.random((2, 200)).astype(np.float32)
step_in = np.array([20], dtype=np.int64)
window_in = np.random.random((32,)).astype(np.float32)
len_in = np.array([64], dtype=np.int64)

start_time = time.time()
np_out = np_stft(numpy_in, step_in[0], window=window_in, frame_length=len_in[0], onesided=True)
print("Numpy:", np_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
kp_out = stft_op.run(numpy_in, step_in, window_in, len_in)[0]
print(f"{stft_op}:", kp_out.shape, time.time() - start_time, "seconds")
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

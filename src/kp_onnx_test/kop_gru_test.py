from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_gru import GruOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())


def f(x):
    return 1 / (1 + np.exp(-x))


def g(x):
    return np.tanh(x)


def _step(X, R, B, W, H_0, num_directions, linear_before_reset, layout):  # type: ignore
    seq_length = X.shape[0]
    hidden_size = H_0.shape[-1]
    batch_size = X.shape[1]

    Y = np.empty([seq_length, num_directions, batch_size, hidden_size])
    h_list = []

    [w_z, w_r, w_h] = np.split(W, 3)
    [r_z, r_r, r_h] = np.split(R, 3)
    [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(B, 6)
    gates_w = np.transpose(np.concatenate((w_z, w_r)))
    gates_r = np.transpose(np.concatenate((r_z, r_r)))
    gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

    H_t = H_0
    for x in np.split(X, X.shape[0], axis=0):
        gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
        z, r = np.split(gates, 2, -1)
        z = f(z)
        r = f(r)
        h_default = g(
            np.dot(x, np.transpose(w_h))
            + np.dot(r * H_t, np.transpose(r_h))
            + w_bh
            + r_bh
        )
        h_linear = g(
            np.dot(x, np.transpose(w_h))
            + r * (np.dot(H_t, np.transpose(r_h)) + r_bh)
            + w_bh
        )
        h = h_linear if linear_before_reset else h_default  # type: ignore
        H = (1 - z) * h + z * H_t
        h_list.append(H)
        H_t = H

    concatenated = np.concatenate(h_list)
    if num_directions == 1:
        Y[:, 0, :, :] = concatenated

    if layout == 0:  # type: ignore
        Y_h = Y[-1]
    else:
        Y = np.transpose(Y, [2, 0, 1, 3])
        Y_h = Y[:, :, -1, :]

    return Y, Y_h


def onnx_gru(
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction=None,
        hidden_size=None,
        layout=None,
        linear_before_reset=None,
):
    # TODO: support overridden attributes.
    num_directions = W.shape[0]
    number_of_gates = 3

    if num_directions == 1:
        R = np.squeeze(R, axis=0)
        W = np.squeeze(W, axis=0)
        if B is not None:
            B = np.squeeze(B, axis=0)
        if sequence_lens is not None:
            sequence_lens = np.squeeze(sequence_lens, axis=0)
        if initial_h is not None:
            initial_h = np.squeeze(initial_h, axis=0)

        hidden_size = R.shape[-1]
        batch_size = X.shape[1]

        X = X if layout == 0 else np.swapaxes(X, 0, 1)
        b = (
            B
            if B is not None
            else np.zeros(2 * number_of_gates * hidden_size, dtype=X.dtype)
        )
        h_0 = (
            initial_h
            if initial_h is not None
            else np.zeros((batch_size, hidden_size), dtype=X.dtype)
        )

        B = b
        H_0 = h_0
    else:
        raise NotImplementedError(f"Unsupported value {num_directions} for num_directions and operator ")

    Y, Y_h = _step(X, R, B, W, H_0, num_directions=num_directions, linear_before_reset=linear_before_reset,
                   layout=layout)
    Y = Y.astype(X.dtype)
    return Y, Y_h.astype(X.dtype)


# ==================== Bidirectional support ====================

def _step_single_direction(X, R, B, W, H_0, linear_before_reset, reverse=False):
    """Process a single direction of GRU (for bidirectional support)."""
    seq_length = X.shape[0]
    hidden_size = H_0.shape[-1]
    batch_size = X.shape[1]

    h_list = []

    [w_z, w_r, w_h] = np.split(W, 3)
    [r_z, r_r, r_h] = np.split(R, 3)
    [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(B, 6)
    gates_w = np.transpose(np.concatenate((w_z, w_r)))
    gates_r = np.transpose(np.concatenate((r_z, r_r)))
    gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

    H_t = H_0

    time_steps = list(range(seq_length))
    if reverse:
        time_steps = time_steps[::-1]

    for t in time_steps:
        x = X[t:t + 1]
        gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
        z, r = np.split(gates, 2, -1)
        z = f(z)
        r = f(r)
        h_default = g(
            np.dot(x, np.transpose(w_h))
            + np.dot(r * H_t, np.transpose(r_h))
            + w_bh
            + r_bh
        )
        h_linear = g(
            np.dot(x, np.transpose(w_h))
            + r * (np.dot(H_t, np.transpose(r_h)) + r_bh)
            + w_bh
        )
        h = h_linear if linear_before_reset else h_default
        H = (1 - z) * h + z * H_t
        h_list.append(H)
        H_t = H

    if reverse:
        h_list = h_list[::-1]

    concatenated = np.concatenate(h_list)
    return concatenated, H_t


def onnx_gru_bidirectional(
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction='bidirectional',
        hidden_size=None,
        layout=0,
        linear_before_reset=0,
):
    """ONNX GRU with bidirectional support."""
    num_directions = W.shape[0]
    number_of_gates = 3
    hidden_size = R.shape[-1]

    # Handle layout: layout=0 -> [seq, batch, input], layout=1 -> [batch, seq, input]
    if layout == 0:
        seq_length = X.shape[0]
        batch_size = X.shape[1]
        X_proc = X
    else:
        batch_size = X.shape[0]
        seq_length = X.shape[1]
        X_proc = np.swapaxes(X, 0, 1)  # -> [seq, batch, input]

    Y = np.empty([seq_length, num_directions, batch_size, hidden_size])

    if num_directions == 1:
        # Single direction (forward or reverse)
        R_d = np.squeeze(R, axis=0)
        W_d = np.squeeze(W, axis=0)
        B_d = np.squeeze(B, axis=0) if B is not None else np.zeros(2 * number_of_gates * hidden_size, dtype=X.dtype)
        H_0 = np.squeeze(initial_h, axis=0) if initial_h is not None else np.zeros((batch_size, hidden_size),
                                                                                   dtype=X.dtype)

        reverse = (direction == 'reverse')
        Y_single, H_final = _step_single_direction(X_proc, R_d, B_d, W_d, H_0, linear_before_reset, reverse=reverse)
        Y[:, 0, :, :] = Y_single

        if layout == 0:
            # Y_h: for forward, last time step; for reverse, first time step (which was processed last)
            if reverse:
                Y_h = Y[0]  # Y_h corresponds to the last processed time step, which is Y[0] for reverse
            else:
                Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])  # -> [batch, seq, num_directions, hidden]
            Y_h = Y[:, :, -1, :]
    else:
        # Bidirectional: num_directions == 2
        B_proc = B if B is not None else np.zeros((num_directions, 2 * number_of_gates * hidden_size), dtype=X.dtype)
        H_0_proc = initial_h if initial_h is not None else np.zeros((num_directions, batch_size, hidden_size),
                                                                    dtype=X.dtype)

        # Forward direction (d=0)
        Y_fwd, H_fwd = _step_single_direction(X_proc, R[0], B_proc[0], W[0], H_0_proc[0], linear_before_reset,
                                              reverse=False)
        Y[:, 0, :, :] = Y_fwd

        # Reverse direction (d=1)
        Y_bwd, H_bwd = _step_single_direction(X_proc, R[1], B_proc[1], W[1], H_0_proc[1], linear_before_reset,
                                              reverse=True)
        Y[:, 1, :, :] = Y_bwd

        # Y_h is [num_directions, batch, hidden] for both layouts
        Y_h = np.zeros([num_directions, batch_size, hidden_size])
        Y_h[0] = Y[-1, 0, :, :]  # Forward: last time step
        Y_h[1] = Y[0, 1, :, :]  # Reverse: first time step

        if layout == 1:
            Y = np.transpose(Y, [2, 0, 1, 3])  # -> [batch, seq, num_directions, hidden]

    Y = Y.astype(X.dtype)
    return Y, Y_h.astype(X.dtype)


# ---------------- Case 1: forward, layout=0, linear=0, 有B有H ----------------
print("Case 1: forward, layout=0, linear=0, 有B有H")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 1

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=B, initial_h=initial_h, layout=0, linear_before_reset=0)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=0)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 2: forward, layout=0, linear=1, 有B有H ----------------
print("Case 2: forward, layout=0, linear=1, 有B有H")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=B, initial_h=initial_h, layout=0, linear_before_reset=1)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=1)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 3: forward, 无B无H (测试可选输入默认值) ----------------
print("Case 3: forward, 无B无H")
seq_length, batch_size, input_size, hidden_size = 4, 2, 3, 5

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=None, initial_h=None, layout=0, linear_before_reset=0)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=0)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, None, None, None)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 4: forward, 有B无H ----------------
print("Case 4: forward, 有B无H")
seq_length, batch_size, input_size, hidden_size = 4, 2, 3, 5

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=B, initial_h=None, layout=0, linear_before_reset=0)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=0)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, None)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 5: forward, 无B有H ----------------
print("Case 5: forward, 无B有H")
seq_length, batch_size, input_size, hidden_size = 4, 2, 3, 5

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=None, initial_h=initial_h, layout=0, linear_before_reset=0)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=0)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, None, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 6: forward, layout=1, linear=0 ----------------
print("Case 6: forward, layout=1, linear=0")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6

X = np.random.random((batch_size, seq_length, input_size)).astype(np.float32)  # layout=1
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=B, initial_h=initial_h, layout=1, linear_before_reset=0)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=1, linear_before_reset=0)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 7: forward, layout=1, linear=1 ----------------
print("Case 7: forward, layout=1, linear=1")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6

X = np.random.random((batch_size, seq_length, input_size)).astype(np.float32)  # layout=1
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru(X, W, R, B=B, initial_h=initial_h, layout=1, linear_before_reset=1)
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=1, linear_before_reset=1)
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 8: reverse, layout=0, linear=0, 无B无H ----------------
print("Case 8: reverse, layout=0, linear=0, 无B无H")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 1

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru_bidirectional(X, W, R, B=None, initial_h=None, layout=0, linear_before_reset=0,
                                      direction='reverse')
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=0, direction='reverse')
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, None, None, None)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 9: reverse, layout=1, linear=1 ----------------
print("Case 9: reverse, layout=1, linear=1")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 1

# layout=1: X shape = [batch, seq, input]
X = np.random.random((batch_size, seq_length, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru_bidirectional(X, W, R, B=B, initial_h=initial_h, layout=1, linear_before_reset=1,
                                      direction='reverse')
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=1, linear_before_reset=1, direction='reverse')
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

## 加上双向，参数数量没有5个的case

# ---------------- Case 10: bidirectional, layout=1, linear=0, 无B无H ----------------
print("Case 10: bidirectional, layout=1, linear=0, 无B无H")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 2

X = np.random.random((batch_size, seq_length, input_size)).astype(np.float32)  # layout=1
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru_bidirectional(X, W, R, B=None, initial_h=None, layout=1, linear_before_reset=0,
                                      direction='bidirectional')
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=1, linear_before_reset=0, direction='bidirectional')
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, None, None, None)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 11: bidirectional, layout=0, linear=0 ----------------
print("Case 11: bidirectional, layout=0, linear=0")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 2

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru_bidirectional(X, W, R, B=B, initial_h=initial_h, layout=0, linear_before_reset=0,
                                      direction='bidirectional')
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=0, direction='bidirectional')
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 12: bidirectional, layout=0, linear=1 ----------------
print("Case 12: bidirectional, layout=0, linear=1")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 2

X = np.random.random((seq_length, batch_size, input_size)).astype(np.float32)
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru_bidirectional(X, W, R, B=B, initial_h=initial_h, layout=0, linear_before_reset=1,
                                      direction='bidirectional')
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=0, linear_before_reset=1, direction='bidirectional')
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

# ---------------- Case 13: bidirectional, layout=1 ----------------
print("Case 13: bidirectional, layout=1")
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
num_directions = 2

X = np.random.random((batch_size, seq_length, input_size)).astype(np.float32)  # layout=1
W = np.random.random((num_directions, 3 * hidden_size, input_size)).astype(np.float32)
R = np.random.random((num_directions, 3 * hidden_size, hidden_size)).astype(np.float32)
B = np.random.random((num_directions, 6 * hidden_size)).astype(np.float32)
initial_h = np.random.random((num_directions, batch_size, hidden_size)).astype(np.float32)

start_time = time.time()
np_Y, np_Y_h = onnx_gru_bidirectional(X, W, R, B=B, initial_h=initial_h, layout=1, linear_before_reset=0,
                                      direction='bidirectional')
print("NumPy:", np_Y.shape, time.time() - start_time, "seconds")
gru_op = GruOp(mgr, hidden_size=hidden_size, layout=1, linear_before_reset=0, direction='bidirectional')
start_time = time.time()
kp_Y, kp_Y_h = gru_op.run(X, W, R, B, None, initial_h)
print(f"{gru_op}:", kp_Y.shape, time.time() - start_time, "seconds")

print("Y allclose:", np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4))
print("Y_h allclose:", np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4))
print("----")

print("All tests completed!")

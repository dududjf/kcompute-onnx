import time
import numpy as np
import kp
from kp_onnx_ssbo.kop_rnn import RnnOp

# Reference implementation based on op_rnn.py
def numpy_rnn(X, W, R, B=None, sequence_lens=None, initial_h=None,
              direction="forward", activations=None, layout=0,
              activation_alpha=None, activation_beta=None):

    if activations is None:
        activations = ["Tanh", "Tanh"]

    num_directions = W.shape[0]
    hidden_size = R.shape[-1]
    batch_size = X.shape[1] if layout == 0 else X.shape[0]
    seq_len = X.shape[0] if layout == 0 else X.shape[1]

    if layout == 1:
        X = np.swapaxes(X, 0, 1) # [batch, seq, input] -> [seq, batch, input]

    if B is None:
        B = np.zeros((num_directions, 2 * hidden_size), dtype=X.dtype)

    if initial_h is None:
        initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=X.dtype)

    Y_list = []
    Y_h_list = []

    for d in range(num_directions):
        # Select weights for direction d
        w_d = W[d] # [hidden, input]
        r_d = R[d] # [hidden, hidden]
        b_d = B[d] # [2*hidden]
        h_0 = initial_h[d] # [batch, hidden]

        act_name = activations[d] if d < len(activations) else "Tanh"
        alpha = activation_alpha[d] if activation_alpha and d < len(activation_alpha) else None
        beta = activation_beta[d] if activation_beta and d < len(activation_beta) else None

        if act_name.lower() == "relu":
            f = lambda x: np.maximum(x, 0)
        elif act_name.lower() == "leakyrelu":
            a = alpha if alpha is not None else 0.01
            f = lambda x: np.where(x >= 0, x, x * a)
        elif act_name.lower() == "thresholdedrelu":
            a = alpha if alpha is not None else 1.0
            f = lambda x: np.where(x > a, x, 0)
        elif act_name.lower() == "sigmoid":
            f = lambda x: 1.0 / (1.0 + np.exp(-x))
        elif act_name.lower() == "hardsigmoid":
            a = alpha if alpha is not None else 0.2
            b = beta if beta is not None else 0.5
            f = lambda x: np.maximum(0.0, np.minimum(1.0, a * x + b))
        elif act_name.lower() == "elu":
            a = alpha if alpha is not None else 1.0
            f = lambda x: np.where(x > 0, x, a * (np.exp(x) - 1.0))
        elif act_name.lower() == "softsign":
            f = lambda x: x / (1.0 + np.abs(x))
        elif act_name.lower() == "softplus":
            f = lambda x: np.log(1.0 + np.exp(x))
        elif act_name.lower() == "scaledtanh":
            a = alpha if alpha is not None else 1.0
            b = beta if beta is not None else 1.0
            f = lambda x: a * np.tanh(b * x)
        elif act_name.lower() == "affine":
            a = alpha if alpha is not None else 1.0
            b = beta if beta is not None else 0.0
            f = lambda x: a * x + b
        else:
            f = np.tanh

        # Direction handling
        is_reverse = (direction == "reverse") or (direction == "bidirectional" and d == 1)

        indices = range(seq_len)
        if is_reverse:
            indices = range(seq_len - 1, -1, -1)

        h_t = h_0
        y_d = np.zeros((seq_len, batch_size, hidden_size), dtype=X.dtype)

        # For Y_h tracking
        last_h_valid = np.zeros((batch_size, hidden_size), dtype=X.dtype)
        last_h_valid[:] = h_0

        for t in indices:
            x_t = X[t] # [batch, input]

            # H_t = f(X_t @ W.T + H_{t-1} @ R.T + B_w + B_r)
            b_w = b_d[:hidden_size]
            b_r = b_d[hidden_size:]

            xw = np.dot(x_t, w_d.T)
            hr = np.dot(h_t, r_d.T)

            val = xw + hr + b_w + b_r
            h_t = f(val)

            # Handle sequence lengths
            if sequence_lens is not None:
                mask_valid = (t < sequence_lens)
                h_t_masked = np.where(mask_valid[:, None], h_t, 0.0)
                y_d[t] = h_t_masked

                if not is_reverse:
                    is_last = (t == sequence_lens - 1)
                    last_h_valid = np.where(is_last[:, None], h_t, last_h_valid)
                else:
                    is_last_reverse = (t == 0)
                    is_valid_step = (t < sequence_lens)
                    should_update = np.logical_and(is_last_reverse, is_valid_step)
                    last_h_valid = np.where(should_update[:, None], h_t, last_h_valid)
            else:
                y_d[t] = h_t
                last_h_valid = h_t

        Y_list.append(y_d)
        Y_h_list.append(last_h_valid)

    # Y: [seq, num_directions, batch, hidden]
    Y = np.stack(Y_list, axis=1)

    if layout == 1:
        # [seq, num_directions, batch, hidden] -> [batch, seq, num_directions, hidden]
        Y = np.transpose(Y, (2, 0, 1, 3))

    # Y_h: [num_directions, batch, hidden]
    Y_h = np.stack(Y_h_list, axis=0)

    return Y, Y_h

def print_test_header(case_num, description):
    """Print test case header"""
    print(f"\nCase {case_num}: {description}")

def run_single_test(manager, direction, layout, activations, X, W, R, B, seq_lens=None, init_h=None, activation_alpha=None, activation_beta=None):
    """Run a single test and return pass/fail"""
    num_directions = W.shape[0]

    t0 = time.time()
    np_Y, np_Y_h = numpy_rnn(X, W, R, B, sequence_lens=seq_lens, initial_h=init_h,
                             direction=direction, activations=activations, layout=layout,
                             activation_alpha=activation_alpha, activation_beta=activation_beta)
    np_time = time.time() - t0

    t0 = time.time()
    rnn_op = RnnOp(manager, direction=direction, layout=layout, activations=activations,
                   activation_alpha=activation_alpha, activation_beta=activation_beta)
    kp_Y, kp_Y_h = rnn_op.run(X, W, R, B, seq_lens, init_h)
    kp_time = time.time() - t0

    print(f"NumPy: {np_Y.shape} {np_time:.4f}s | RnnOp: {kp_Y.shape} {kp_time:.4f}s")

    max_err_Y = np.abs(np_Y - kp_Y).max()
    max_err_Y_h = np.abs(np_Y_h - kp_Y_h).max()
    passed = np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4)

    print(f"Max error Y: {max_err_Y:.6e} | Max error Y_h: {max_err_Y_h:.6e} | All close: {passed}")
    return passed

def run_test():
    manager = kp.Manager()

    # Common parameters
    seq_len, batch, input_size, hidden_size = 5, 2, 4, 3
    X_layout0 = np.random.uniform(-1, 1, (seq_len, batch, input_size)).astype(np.float32)
    X_layout1 = np.swapaxes(X_layout0, 0, 1).copy()



    # Case 1: Forward with default Tanh
    print_test_header(1, "Forward (Default Tanh, Layout 0)")
    rnn_op = RnnOp(manager)
    print(f"repr: {repr(rnn_op)}")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Tanh"], X_layout0, W, R, B)

    # Case 2: Reverse with default Tanh
    print_test_header(2, "Reverse (Default Tanh, Layout 0)")
    run_single_test(manager, "reverse", 0, ["Tanh"], X_layout0, W, R, B)

    # Case 3: Bidirectional with default Tanh
    print_test_header(3, "Bidirectional (Default Tanh, Layout 0)")
    num_directions = 2
    W = np.random.uniform(-1, 1, (num_directions, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (num_directions, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (num_directions, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "bidirectional", 0, ["Tanh", "Tanh"], X_layout0, W, R, B)

    # Case 4: Forward with Layout 1
    print_test_header(4, "Forward (Layout 1)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 1, ["Tanh"], X_layout1, W, R, B)

    # Case 5: Reverse with Layout 1
    print_test_header(5, "Reverse (Layout 1)")
    run_single_test(manager, "reverse", 1, ["Tanh"], X_layout1, W, R, B)

    # Case 6: Bidirectional with Layout 1
    print_test_header(6, "Bidirectional (Layout 1)")
    num_directions = 2
    W = np.random.uniform(-1, 1, (num_directions, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (num_directions, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (num_directions, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "bidirectional", 1, ["Tanh", "Tanh"], X_layout1, W, R, B)

    # Case 7: Forward + No Bias
    print_test_header(7, "Forward (No Bias)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Tanh"], X_layout0, W, R, None)

    # Case 8: Reverse + No Bias
    print_test_header(8, "Reverse (No Bias)")
    run_single_test(manager, "reverse", 0, ["Tanh"], X_layout0, W, R, None)

    # Case 9: Forward + Sequence Lens
    print_test_header(9, "Forward (Sequence Lens)")
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    seq_lens = np.array([seq_len, 3], dtype=np.int32)
    run_single_test(manager, "forward", 0, ["Tanh"], X_layout0, W, R, B, seq_lens=seq_lens)

    # Case 10: Reverse + Layout 1
    print_test_header(10, "Reverse (Layout 1)")
    run_single_test(manager, "reverse", 1, ["Tanh"], X_layout1, W, R, B)

    # Case 11: Forward + Initial H
    print_test_header(11, "Forward (Initial H)")
    init_h = np.random.uniform(-1, 1, (1, batch, hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Tanh"], X_layout0, W, R, B, init_h=init_h)

    # Case 12: Reverse + Initial H
    print_test_header(12, "Reverse (Initial H)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    init_h = np.random.uniform(-1, 1, (1, batch, hidden_size)).astype(np.float32)
    run_single_test(manager, "reverse", 0, ["Tanh"], X_layout0, W, R, B, init_h=init_h)

    # Case 13: Bidirectional + Initial H
    print_test_header(13, "Bidirectional (Initial H)")
    num_directions = 2
    W = np.random.uniform(-1, 1, (num_directions, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (num_directions, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (num_directions, 2 * hidden_size)).astype(np.float32)
    init_h = np.random.uniform(-1, 1, (num_directions, batch, hidden_size)).astype(np.float32)
    run_single_test(manager, "bidirectional", 0, ["Tanh", "Tanh"], X_layout0, W, R, B, init_h=init_h)

    # Case 14: Bidirectional + Layout 1
    print_test_header(14, "Bidirectional (Layout 1)")
    num_directions = 2
    W = np.random.uniform(-1, 1, (num_directions, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (num_directions, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (num_directions, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "bidirectional", 1, ["Tanh", "Tanh"], X_layout1, W, R, B)

    # Case 15: Forward (Relu)
    print_test_header(15, "Forward (Relu)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Relu"], X_layout0, W, R, B)

    # Case 16: Forward (LeakyRelu)
    print_test_header(16, "Forward (LeakyRelu)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["LeakyRelu"], X_layout0, W, R, B,
                      activation_alpha=[0.5])

    # Case 17: Forward (ThresholdedRelu)
    print_test_header(17, "Forward (ThresholdedRelu)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["ThresholdedRelu"], X_layout0, W, R, B,
                      activation_alpha=[0.5])

    # Case 18: Forward (Sigmoid)
    print_test_header(18, "Forward (Sigmoid)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Sigmoid"], X_layout0, W, R, B)

    # Case 19: Forward (HardSigmoid)
    print_test_header(19, "Forward (HardSigmoid)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["HardSigmoid"], X_layout0, W, R, B,
                      activation_alpha=[0.2], activation_beta=[0.5])

    # Case 20: Forward (Elu)
    print_test_header(20, "Forward (Elu)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Elu"], X_layout0, W, R, B,
                      activation_alpha=[0.5])

    # Case 21: Forward (Softsign)
    print_test_header(21, "Forward (Softsign)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Softsign"], X_layout0, W, R, B)

    # Case 22: Forward (Softplus)
    print_test_header(22, "Forward (Softplus)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Softplus"], X_layout0, W, R, B)

    # Case 23: Forward (ScaledTanh)
    print_test_header(23, "Forward (ScaledTanh)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["ScaledTanh"], X_layout0, W, R, B,
                      activation_alpha=[0.5], activation_beta=[0.2])

    # Case 24: Forward (Affine)
    print_test_header(24, "Forward (Affine)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, ["Affine"], X_layout0, W, R, B,
                      activation_alpha=[0.5], activation_beta=[0.2])

    # Case 25: Bidirectional with Mixed Activations (Tanh and Relu)
    print_test_header(25, "Bidirectional Mixed Activations (Tanh/Relu)")
    num_directions = 2
    W = np.random.uniform(-1, 1, (num_directions, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (num_directions, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (num_directions, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "bidirectional", 0, ["Tanh", "Relu"], X_layout0, W, R, B)

    # Case 26: Test with None parameters (to cover default value branches)
    print_test_header(26, "Forward (None activations, activation_alpha, activation_beta)")
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "forward", 0, None, X_layout0, W, R, B, activation_alpha=None, activation_beta=None)

    # Case 27: Bidirectional with only 1 activation (to cover while loop filling defaults)
    print_test_header(27, "Bidirectional (Only 1 activation provided)")
    num_directions = 2
    W = np.random.uniform(-1, 1, (num_directions, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (num_directions, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (num_directions, 2 * hidden_size)).astype(np.float32)
    run_single_test(manager, "bidirectional", 0, ["Relu"], X_layout0, W, R, B)

    # Case 28: Test set_attributes with explicit None to cover else branches
    print_test_header(28, "Test set_attributes with explicit None values")
    rnn_op = RnnOp(manager, direction="forward", layout=0, hidden_size=hidden_size)
    # 现在调用set_attributes并传入None来触发else分支
    rnn_op.set_attributes(activations=None, activation_alpha=None, activation_beta=None)
    W = np.random.uniform(-1, 1, (1, hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-1, 1, (1, hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-1, 1, (1, 2 * hidden_size)).astype(np.float32)
    kp_Y, kp_Y_h = rnn_op.run(X_layout0, W, R, B, None, None)
    np_Y, np_Y_h = numpy_rnn(X_layout0, W, R, B, direction="forward", layout=0)
    passed = np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4)
    print(f"Max error Y: {np.abs(np_Y - kp_Y).max():.6e} | All close: {passed}")

if __name__ == "__main__":
    run_test()


import time
import numpy as np
import kp
from kp_onnx_ssbo.kop_lstm import LstmOp

# Reference implementation based on op_lstm.py
def numpy_lstm(X, W, R, B=None, sequence_lens=None, initial_h=None, initial_c=None, P=None,
               direction="forward", activations=None, layout=0,
               activation_alpha=None, activation_beta=None, clip=None, input_forget=0):
    """
    LSTM reference implementation
    """
    if activations is None:
        activations = ["Sigmoid", "Tanh", "Tanh"]

    num_directions = W.shape[0]
    hidden_size = R.shape[-1]
    batch_size = X.shape[1] if layout == 0 else X.shape[0]
    seq_len = X.shape[0] if layout == 0 else X.shape[1]

    if layout == 1:
        X = np.swapaxes(X, 0, 1)  # [batch, seq, input] -> [seq, batch, input]

    if B is None:
        B = np.zeros((num_directions, 8 * hidden_size), dtype=X.dtype)

    if P is None:
        P = np.zeros((num_directions, 3 * hidden_size), dtype=X.dtype)

    if initial_h is None:
        initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=X.dtype)

    if initial_c is None:
        initial_c = np.zeros((num_directions, batch_size, hidden_size), dtype=X.dtype)

    Y_list = []
    Y_h_list = []
    Y_c_list = []

    # Track consumed alpha/beta indices globally for the test run
    alpha_consumption_idx = 0
    beta_consumption_idx = 0

    for d in range(num_directions):
        # Select weights for direction d
        w_d = W[d]  # [4*hidden, input]
        r_d = R[d]  # [4*hidden, hidden]
        b_d = B[d]  # [8*hidden]
        p_d = P[d]  # [3*hidden]
        h_0 = initial_h[d]  # [batch, hidden]
        c_0 = initial_c[d]  # [batch, hidden]

        # Parse activations for this direction
        f_name = activations[d * 3] if d * 3 < len(activations) else "Sigmoid"
        g_name = activations[d * 3 + 1] if d * 3 + 1 < len(activations) else "Tanh"
        h_name = activations[d * 3 + 2] if d * 3 + 2 < len(activations) else "Tanh"

        def create_activation(name, alpha_val=None, beta_val=None):
            name_lower = name.lower()
            if name_lower == "relu":
                return lambda x: np.maximum(x, 0)
            elif name_lower == "leakyrelu":
                a = alpha_val if alpha_val is not None else 0.01
                return lambda x: np.where(x >= 0, x, x * a)
            elif name_lower == "thresholdedrelu":
                a = alpha_val if alpha_val is not None else 1.0
                return lambda x: np.where(x > a, x, 0)
            elif name_lower == "sigmoid":
                return lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
            elif name_lower == "hardsigmoid":
                a = alpha_val if alpha_val is not None else 0.2
                b = beta_val if beta_val is not None else 0.5
                return lambda x: np.maximum(0.0, np.minimum(1.0, a * x + b))
            elif name_lower == "elu":
                a = alpha_val if alpha_val is not None else 1.0
                return lambda x: np.where(x > 0, x, a * (np.exp(x) - 1.0))
            elif name_lower == "softsign":
                return lambda x: x / (1.0 + np.abs(x))
            elif name_lower == "softplus":
                return lambda x: np.log(1.0 + np.exp(np.clip(x, -50, 50)))
            elif name_lower == "scaledtanh":
                a = alpha_val if alpha_val is not None else 1.0
                b = beta_val if beta_val is not None else 1.0
                return lambda x: a * np.tanh(b * x)
            elif name_lower == "affine":
                a = alpha_val if alpha_val is not None else 1.0
                b = beta_val if beta_val is not None else 0.0
                return lambda x: a * x + b
            else:
                return np.tanh

        def get_params(act_name):
            nonlocal alpha_consumption_idx, beta_consumption_idx
            act_lower = act_name.lower()
            alpha = None
            beta = None

            uses_alpha = act_lower in ["leakyrelu", "thresholdedrelu", "elu", "hardsigmoid", "scaledtanh", "affine"]
            uses_beta = act_lower in ["hardsigmoid", "scaledtanh", "affine"]

            if uses_alpha and activation_alpha and alpha_consumption_idx < len(activation_alpha):
                alpha = activation_alpha[alpha_consumption_idx]
                alpha_consumption_idx += 1

            if uses_beta and activation_beta and beta_consumption_idx < len(activation_beta):
                beta = activation_beta[beta_consumption_idx]
                beta_consumption_idx += 1

            return alpha, beta

        f_alpha, f_beta = get_params(f_name)
        g_alpha, g_beta = get_params(g_name)
        h_alpha, h_beta = get_params(h_name)

        f_func = create_activation(f_name, f_alpha, f_beta)
        g_func = create_activation(g_name, g_alpha, g_beta)
        h_func = create_activation(h_name, h_alpha, h_beta)

        # Direction handling
        is_reverse = (direction == "reverse") or (direction == "bidirectional" and d == 1)

        indices = range(seq_len)
        if is_reverse:
            indices = range(seq_len - 1, -1, -1)

        h_t = h_0
        c_t = c_0
        y_d = np.zeros((seq_len, batch_size, hidden_size), dtype=X.dtype)

        # For Y_h tracking
        last_h_valid = np.zeros((batch_size, hidden_size), dtype=X.dtype)
        last_h_valid[:] = h_0
        last_c_valid = np.zeros((batch_size, hidden_size), dtype=X.dtype)
        last_c_valid[:] = c_0

        for t in indices:
            x_t = X[t]  # [batch, input]

            # Split weights and biases
            w_i, w_o, w_f, w_c = np.split(w_d, 4, axis=0)
            r_i, r_o, r_f, r_c = np.split(r_d, 4, axis=0)
            b_wi, b_wo, b_wf, b_wc, b_ri, b_ro, b_rf, b_rc = np.split(b_d, 8)
            p_i, p_o, p_f = np.split(p_d, 3)

            # Gates calculation
            it = np.dot(x_t, w_i.T) + np.dot(h_t, r_i.T) + b_wi + b_ri + p_i * c_t
            ft = np.dot(x_t, w_f.T) + np.dot(h_t, r_f.T) + b_wf + b_rf + p_f * c_t
            ct_tilde = np.dot(x_t, w_c.T) + np.dot(h_t, r_c.T) + b_wc + b_rc
            ot = np.dot(x_t, w_o.T) + np.dot(h_t, r_o.T) + b_wo + b_ro

            # Clip if specified
            if clip is not None:
                it = np.clip(it, -clip, clip)
                ft = np.clip(ft, -clip, clip)
                ct_tilde = np.clip(ct_tilde, -clip, clip)
                ot = np.clip(ot, -clip, clip)

            # Apply activations
            it = f_func(it)
            ft = f_func(ft)
            ct_tilde = g_func(ct_tilde)

            # Cell state update
            c_t = ft * c_t + it * ct_tilde

            # Output gate with peephole
            ot = f_func(ot + p_o * c_t)

            # Hidden state
            h_t = ot * h_func(c_t)

            # Handle sequence lengths
            if sequence_lens is not None:
                mask_valid = (t < sequence_lens)
                h_t_masked = np.where(mask_valid[:, None], h_t, 0.0)
                c_t_masked = np.where(mask_valid[:, None], c_t, 0.0)
                y_d[t] = h_t_masked
                c_t = c_t_masked

                if not is_reverse:
                    is_last = (t == sequence_lens - 1)
                    last_h_valid = np.where(is_last[:, None], h_t, last_h_valid)
                    last_c_valid = np.where(is_last[:, None], c_t, last_c_valid)
                else:
                    is_last_reverse = (t == 0)
                    is_valid_step = (t < sequence_lens)
                    should_update = np.logical_and(is_last_reverse, is_valid_step)
                    last_h_valid = np.where(should_update[:, None], h_t, last_h_valid)
                    last_c_valid = np.where(should_update[:, None], c_t, last_c_valid)
            else:
                y_d[t] = h_t
                last_h_valid = h_t
                last_c_valid = c_t

        Y_list.append(y_d)
        Y_h_list.append(last_h_valid)
        Y_c_list.append(last_c_valid)

    # Y: [seq, num_directions, batch, hidden]
    Y = np.stack(Y_list, axis=1)

    if layout == 1:
        # [seq, num_directions, batch, hidden] -> [batch, seq, num_directions, hidden]
        Y = np.transpose(Y, (2, 0, 1, 3))

    # Y_h: [num_directions, batch, hidden]
    Y_h = np.stack(Y_h_list, axis=0)
    Y_c = np.stack(Y_c_list, axis=0)

    return Y, Y_h, Y_c


def print_test_header(case_num, description):
    """Print test case header"""
    print(f"\nCase {case_num}: {description}")


def run_single_test(manager, direction, layout, activations, X, W, R, B=None, seq_lens=None,
                    init_h=None, init_c=None, P=None, activation_alpha=None, activation_beta=None,
                    clip=None, input_forget=0):
    """Run a single test and return pass/fail"""
    num_directions = W.shape[0]

    t0 = time.time()
    np_Y, np_Y_h, np_Y_c = numpy_lstm(X, W, R, B, sequence_lens=seq_lens, initial_h=init_h,
                                       initial_c=init_c, P=P, direction=direction,
                                       activations=activations, layout=layout,
                                       activation_alpha=activation_alpha,
                                       activation_beta=activation_beta,
                                       clip=clip, input_forget=input_forget)
    np_time = time.time() - t0

    t0 = time.time()
    lstm_op = LstmOp(manager, direction=direction, layout=layout, activations=activations,
                     activation_alpha=activation_alpha, activation_beta=activation_beta,
                     clip=clip, input_forget=input_forget)
    kp_Y, kp_Y_h, kp_Y_c = lstm_op.run(X, W, R, B, seq_lens, init_h, init_c, P)
    kp_time = time.time() - t0

    print(f"NumPy: {np_Y.shape} {np_time:.4f}s | LstmOp: {kp_Y.shape} {kp_time:.4f}s")

    max_err_Y = np.abs(np_Y - kp_Y).max()
    max_err_Y_h = np.abs(np_Y_h - kp_Y_h).max()
    max_err_Y_c = np.abs(np_Y_c - kp_Y_c).max()
    passed = np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4) and \
             np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4) and \
             np.allclose(np_Y_c, kp_Y_c, rtol=1e-4, atol=1e-4)

    print(f"Max error Y: {max_err_Y:.6e} | Max error Y_h: {max_err_Y_h:.6e} | Max error Y_c: {max_err_Y_c:.6e} | All close: {passed}")
    return passed


def run_test():
    manager = kp.Manager()
    lstm_op = LstmOp(manager)
    print(f"repr: {repr(lstm_op)}")

    # Common parameters
    seq_len, batch, input_size, hidden_size = 5, 2, 4, 3
    # Use smaller weights for numerical stability in float32
    X_layout0 = np.random.uniform(-0.01, 0.01, (seq_len, batch, input_size)).astype(np.float32)
    X_layout1 = np.swapaxes(X_layout0, 0, 1).copy()

    passed_count = 0
    total_count = 0

    # Case 1: Forward with default activations (Sigmoid, Tanh, Tanh)
    print_test_header(1, "Forward (Default Sigmoid/Tanh/Tanh, Layout 0)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1

    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B):
        passed_count += 1

    # Case 2: Reverse with default activations
    print_test_header(2, "Reverse (Default Sigmoid/Tanh/Tanh, Layout 0)")
    total_count += 1
    if run_single_test(manager, "reverse", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B):
        passed_count += 1

    # Case 3: Bidirectional with default activations
    print_test_header(3, "Bidirectional (Default Sigmoid/Tanh/Tanh, Layout 0)")
    num_directions = 2
    W = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (num_directions, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "bidirectional", 0, ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"],
                      X_layout0, W, R, B):
        passed_count += 1

    # Case 4: Forward with Layout 1
    print_test_header(4, "Forward (Layout 1)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 1, ["Sigmoid", "Tanh", "Tanh"], X_layout1, W, R, B):
        passed_count += 1

    # Case 5: Reverse with Layout 1
    print_test_header(5, "Reverse (Layout 1)")
    total_count += 1
    if run_single_test(manager, "reverse", 1, ["Sigmoid", "Tanh", "Tanh"], X_layout1, W, R, B):
        passed_count += 1

    # Case 6: Bidirectional with Layout 1
    print_test_header(6, "Bidirectional (Layout 1)")
    num_directions = 2
    W = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (num_directions, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "bidirectional", 1, ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"],
                      X_layout1, W, R, B):
        passed_count += 1

    # Case 7: Forward + No Bias
    print_test_header(7, "Forward (No Bias)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, None):
        passed_count += 1

    # Case 8: Reverse + No Bias
    print_test_header(8, "Reverse (No Bias)")
    total_count += 1
    if run_single_test(manager, "reverse", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, None):
        passed_count += 1

    # Case 9: Forward + Sequence Lens
    print_test_header(9, "Forward (Sequence Lens)")
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    seq_lens = np.array([seq_len, 3], dtype=np.int32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B, seq_lens=seq_lens):
        passed_count += 1

    # Case 10: Reverse + Sequence Lens
    print_test_header(10, "Reverse (Sequence Lens)")
    total_count += 1
    if run_single_test(manager, "reverse", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B, seq_lens=seq_lens):
        passed_count += 1

    # Case 11: Forward + Initial H and C
    print_test_header(11, "Forward (Initial H and C)")
    init_h = np.random.uniform(-0.01, 0.01, (1, batch, hidden_size)).astype(np.float32)
    init_c = np.random.uniform(-0.01, 0.1, (1, batch, hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B,
                      init_h=init_h, init_c=init_c):
        passed_count += 1

    # Case 12: Reverse + Initial H and C
    print_test_header(12, "Reverse (Initial H and C)")
    total_count += 1
    if run_single_test(manager, "reverse", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B,
                      init_h=init_h, init_c=init_c):
        passed_count += 1

    # Case 13: Bidirectional + Initial H and C
    print_test_header(13, "Bidirectional (Initial H and C)")
    num_directions = 2
    W = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (num_directions, 8 * hidden_size)).astype(np.float32)
    init_h = np.random.uniform(-0.01, 0.01, (num_directions, batch, hidden_size)).astype(np.float32)
    init_c = np.random.uniform(-0.01, 0.01, (num_directions, batch, hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "bidirectional", 0, ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"],
                      X_layout0, W, R, B, init_h=init_h, init_c=init_c):
        passed_count += 1

    # Case 14: Forward + Peepholes
    print_test_header(14, "Forward (Peepholes)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    P = np.random.uniform(-0.01, 0.01, (1, 3 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B, P=P):
        passed_count += 1

    # Case 15: Reverse + Peepholes
    print_test_header(15, "Reverse (Peepholes)")
    total_count += 1
    if run_single_test(manager, "reverse", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B, P=P):
        passed_count += 1

    # Case 16: Bidirectional + Peepholes
    print_test_header(16, "Bidirectional (Peepholes)")
    num_directions = 2
    W = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (num_directions, 8 * hidden_size)).astype(np.float32)
    P = np.random.uniform(-0.01, 0.01, (num_directions, 3 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "bidirectional", 0, ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"],
                      X_layout0, W, R, B, P=P):
        passed_count += 1

    # Case 17: Forward + Clip
    print_test_header(17, "Forward (Clip)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B, clip=5.0):
        passed_count += 1

    # Case 18: Reverse + Clip
    print_test_header(18, "Reverse (Clip)")
    total_count += 1
    if run_single_test(manager, "reverse", 0, ["Sigmoid", "Tanh", "Tanh"], X_layout0, W, R, B, clip=5.0):
        passed_count += 1

    # Case 19: Forward (Relu/Tanh/Tanh)
    print_test_header(19, "Forward (Relu/Tanh/Tanh)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Relu", "Tanh", "Tanh"], X_layout0, W, R, B):
        passed_count += 1

    # Case 20: Forward (Sigmoid/Relu/Tanh)
    print_test_header(20, "Forward (Sigmoid/Relu/Tanh)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Relu", "Tanh"], X_layout0, W, R, B):
        passed_count += 1

    # Case 21: Forward (Sigmoid/Tanh/Relu)
    print_test_header(21, "Forward (Sigmoid/Tanh/Relu)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["Sigmoid", "Tanh", "Relu"], X_layout0, W, R, B):
        passed_count += 1

    # Case 22: Forward (HardSigmoid/Tanh/Tanh)
    print_test_header(22, "Forward (HardSigmoid/Tanh/Tanh)")
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "forward", 0, ["HardSigmoid", "Tanh", "Tanh"], X_layout0, W, R, B,
                      activation_alpha=[0.2], activation_beta=[0.5]):
        passed_count += 1

    # Case 23: Bidirectional Mixed Activations
    print_test_header(23, "Bidirectional Mixed Activations")
    num_directions = 2
    W = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (num_directions, 8 * hidden_size)).astype(np.float32)
    total_count += 1
    if run_single_test(manager, "bidirectional", 0,
                      ["Sigmoid", "Tanh", "Tanh", "Relu", "Tanh", "Relu"],
                      X_layout0, W, R, B):
        passed_count += 1

    # Case 24: Complex (Bidirectional + Layout1 + SeqLens + InitH/C + Peepholes + Clip)
    print_test_header(24, "Complex (Bidirectional + Layout1 + SeqLens + InitH/C + Peepholes + Clip)")
    num_directions = 2
    W = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (num_directions, 8 * hidden_size)).astype(np.float32)
    P = np.random.uniform(-0.01, 0.01, (num_directions, 3 * hidden_size)).astype(np.float32)
    init_h = np.random.uniform(-0.01, 0.01, (num_directions, batch, hidden_size)).astype(np.float32)
    init_c = np.random.uniform(-0.01, 0.01, (num_directions, batch, hidden_size)).astype(np.float32)
    seq_lens = np.array([seq_len, 3], dtype=np.int32)
    total_count += 1
    if run_single_test(manager, "bidirectional", 1,
                      ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"],
                      X_layout1, W, R, B, seq_lens=seq_lens, init_h=init_h, init_c=init_c, P=P, clip=3.0):
        passed_count += 1

    # Case 25: Test set_attributes with explicit None to cover default value branches
    print_test_header(25, "Test set_attributes with None values (coverage test)")
    lstm_op = LstmOp(manager, direction="forward", layout=0, hidden_size=hidden_size)
    # 调用set_attributes并传入None来触发默认值分支
    lstm_op.set_attributes(activations=None, activation_alpha=None, activation_beta=None)
    W = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-0.01, 0.01, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.random.uniform(-0.01, 0.01, (1, 8 * hidden_size)).astype(np.float32)
    kp_Y, kp_Y_h, kp_Y_c = lstm_op.run(X_layout0, W, R, B, None, None, None, None)
    np_Y, np_Y_h, np_Y_c = numpy_lstm(X_layout0, W, R, B, direction="forward", layout=0)
    passed = np.allclose(np_Y, kp_Y, rtol=1e-4, atol=1e-4) and \
             np.allclose(np_Y_h, kp_Y_h, rtol=1e-4, atol=1e-4) and \
             np.allclose(np_Y_c, kp_Y_c, rtol=1e-4, atol=1e-4)
    print(f"Max error Y: {np.abs(np_Y - kp_Y).max():.6e} | All close: {passed}")
    total_count += 1
    if passed:
        passed_count += 1

    # Print summary
    print("\n" + "="*60)
    print(f"SUMMARY: {passed_count}/{total_count} tests passed")
    print("="*60)

    if passed_count == total_count:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total_count - passed_count} test(s) failed")


if __name__ == "__main__":
    run_test()


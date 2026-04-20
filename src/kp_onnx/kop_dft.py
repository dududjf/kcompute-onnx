import kp
import numpy as np
import math
from .shader_utils import compile_source

DEFAULT_AXIS = -2


class DFTOp:

    def __init__(self, manager: kp.Manager, inverse=0, onesided=0): # 根据ref改成bool
        self.inverse = inverse
        self.onesided = onesided
        self.manager = manager
        self.dft_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float pre_size_f = 0.0;
layout(constant_id = 1) const float input_n_f = 0.0;
layout(constant_id = 2) const float dft_length_f = 0.0;
layout(constant_id = 3) const float output_n_f = 0.0;
layout(constant_id = 4) const float post_size_f = 0.0;
layout(constant_id = 5) const float norm_f = 0.0;
layout(constant_id = 6) const float angle_base_f = 0.0;
layout(constant_id = 7) const float in_idx_step_f = 0.0;
layout(constant_id = 8) const float is_real_input_f = 0.0;

void main() {
    uint pre_idx = gl_GlobalInvocationID.x;
    uint k = gl_GlobalInvocationID.y;
    uint post_idx = gl_GlobalInvocationID.z;

    uint pre_size = uint(pre_size_f);
    uint input_n = uint(input_n_f);
    uint dft_length = uint(dft_length_f);
    uint output_n = uint(output_n_f);
    uint post_size = uint(post_size_f);
    float norm = norm_f;
    float angle_base = angle_base_f;
    uint in_idx_step = uint(in_idx_step_f);
    bool is_real_input = is_real_input_f > 0.5;

    float real_sum = 0.0;
    float imag_sum = 0.0;
    float real_c = 0.0;
    float imag_c = 0.0;

    float angle_increment = angle_base * float(k);
    float cos_inc = cos(angle_increment);
    float sin_inc = sin(angle_increment);

    uint in_idx_base = pre_idx * input_n * post_size + post_idx;
    uint in_idx = is_real_input ? in_idx_base : in_idx_base * 2;
    uint in_step = is_real_input ? post_size : in_idx_step;

    float cos_angle = 1.0;
    float sin_angle = 0.0;

    uint loop_n = min(input_n, dft_length);
    for (uint n = 0; n < loop_n; ++n) {
        float x_real = in_buf[in_idx];
        float x_imag = is_real_input ? 0.0 : in_buf[in_idx + 1];

        float real_term = x_real * cos_angle - x_imag * sin_angle;
        float imag_term = x_real * sin_angle + x_imag * cos_angle;

        float real_y = real_term - real_c;
        float real_t = real_sum + real_y;
        real_c = (real_t - real_sum) - real_y;
        real_sum = real_t;

        float imag_y = imag_term - imag_c;
        float imag_t = imag_sum + imag_y;
        imag_c = (imag_t - imag_sum) - imag_y;
        imag_sum = imag_t;

        in_idx += in_step;

        float tmp_cos = cos_angle * cos_inc - sin_angle * sin_inc;
        sin_angle = sin_angle * cos_inc + cos_angle * sin_inc;
        cos_angle = tmp_cos;
    }

    uint out_idx = (pre_idx * output_n * post_size + k * post_size + post_idx) * 2;
    out_buf[out_idx] = real_sum * norm;
    out_buf[out_idx + 1] = imag_sum * norm;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"DFTOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            if numpy_in.size > 0:
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))
            else:
                input_tensors.append((None, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors if t[0] is not None]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            if tensor is not None:
                del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, in_shape = input_tensors[0]

        # Handle optional dft_length input
        dft_length = None
        axis = DEFAULT_AXIS

        if len(input_tensors) > 1:
            dft_length_tensor, dft_length_shape = input_tensors[1]
            if dft_length_tensor is not None and dft_length_shape and dft_length_shape[0] > 0:
                dft_length_data = dft_length_tensor.data()
                dft_length = int(dft_length_data[0])

        if len(input_tensors) > 2:
            axis_tensor, axis_shape = input_tensors[2]
            if axis_tensor is not None and axis_shape and axis_shape[0] > 0:
                axis_data = axis_tensor.data()
                axis = int(axis_data[0])

        # Convert axis to positive
        axis = axis + len(in_shape) if axis < 0 else axis

        assert 0 <= axis < len(in_shape), f"axis {axis} out of range for shape {in_shape}"
        assert in_shape[-1] in (1, 2), f"Last dimension must be 1 (real) or 2 (complex), got {in_shape[-1]}"

        # Check if input is real (last dimension is 1) or complex (last dimension is 2)
        is_real_input = in_shape[-1] == 1

        # Determine DFT parameters
        input_n = in_shape[axis]
        dft_length = dft_length if dft_length is not None else input_n
        actual_output_n = (dft_length >> 1) + 1 if self.onesided else dft_length

        # Calculate dimensions for workgroup
        pre_size = int(np.prod(in_shape[:axis]))
        post_size = int(np.prod(in_shape[axis+1:-1]))

        # Create output tensor (always complex, last dimension is 2)
        out_shape = in_shape[:axis] + [actual_output_n] + in_shape[axis+1:-1] + [2]
        tensor_out = self.manager.tensor(np.zeros(int(np.prod(out_shape)), dtype=np.float32))
        updated_tensors.append(tensor_out)

        # Compute constants on CPU
        sign = 1.0 if self.inverse else -1.0
        norm = (1.0 / dft_length) if self.inverse else 1.0
        angle_base = sign * 2.0 * math.pi / dft_length
        in_idx_step = post_size * 2

        # Execute DFT computation
        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                self.dft_shader,
                (pre_size, actual_output_n, post_size),
                [
                    pre_size,
                    input_n,
                    dft_length,
                    actual_output_n,
                    post_size,
                    norm,
                    angle_base,
                    in_idx_step,
                    1.0 if is_real_input else 0.0
                ],
                []
            )
        )

        return [(tensor_out, out_shape)]
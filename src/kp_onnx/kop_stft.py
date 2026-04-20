from __future__ import annotations
import kp
import numpy as np
from .shader_utils import compile_source


class STFTOp:
    def __init__(self, manager: kp.Manager, onesided: int = 1):
        self.manager = manager
        self.onesided = onesided
        self.compiled_shader = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer buf_in_x      { float in_x[]; };
layout (binding = 1) buffer buf_in_window { float in_window[]; };
layout (binding = 2) buffer buf_out_result { float out_result[]; };

layout (constant_id = 0) const float x_len_f = 0;      // len(x)
layout (constant_id = 1) const float hop_length_f = 0; // hop_length
layout (constant_id = 2) const float window_size_f = 0;// window_size
layout (constant_id = 3) const float fft_length_f = 0; // fft_length
layout (constant_id = 4) const float n_frames_f = 0;   // n_frames
layout (constant_id = 5) const float n_freqs_f = 0;    // 输出的频率维度
layout (constant_id = 6) const float use_window_f = 0; // 标记是否传入了window

#define PI 3.14159265358979323846

void main()
{
    // ONNX: for fs in range(n_frames):
    uint fs = gl_GlobalInvocationID.x; 

    // ONNX _dft 内部逻辑的频率索引 k
    uint k  = gl_GlobalInvocationID.y; 

    uint batch_idx = gl_GlobalInvocationID.z;

    uint n_frames = uint(n_frames_f);
    uint n_freqs  = uint(n_freqs_f);

    uint x_len = uint(x_len_f);
    uint hop_length = uint(hop_length_f);
    uint window_size = uint(window_size_f);
    uint fft_length = uint(fft_length_f);
    uint use_window = uint(use_window_f);

    // 计算 result 的内存索引
    uint stride_freq = 2; // 复数 (real, imag)
    uint stride_frame = n_freqs * stride_freq;
    uint stride_batch = n_frames * stride_frame;

    uint out_idx = (batch_idx * stride_batch) + (fs * stride_frame) + (k * stride_freq);

    // 步骤 1: 切片逻辑
    // 对应 ONNX: begin = fs * hop_length
    // 对应 ONNX: end = begin + window_size （逻辑上存在，循环中使用 n 控制）
    uint begin = (batch_idx * x_len) + (fs * hop_length);

    float sum_r = 0.0;
    float sum_i = 0.0;

    uint phase_acc = 0;

    // 对应 ONNX: 遍历切片内的每一个点进行处理
    // 这里的循环相当于 Python 代码中构建 seq 和 _dft 计算的结合
    for (uint n = 0; n < window_size; ++n) {

        // 步骤 2: 获取切片数据并 Padding
        // 对应 ONNX: sliced_x = _slice(...) 
        // 对应 ONNX: pad_sliced_x = _concat(sliced_x, cst, ...)
        float pad_sliced_x = 0.0; // 默认为 0 (Padding)

        uint current_idx = begin + n;

        // 只有在范围内才取值，否则保持 0 (实现 _concat padding 逻辑)
        if (current_idx < (batch_idx + 1) * x_len) {
            pad_sliced_x = in_x[current_idx];
        }

        // 步骤 3: 加窗
        // 对应 ONNX: weighted_new_x = new_x * weights
        // ONNX 中 new_x 是由 pad_sliced_x 组成的序列
        float weighted_new_x = pad_sliced_x;

        if (use_window == 1) {
            weighted_new_x *= in_window[n];
        }

        // 步骤 4: 执行 DFT
        // 对应 ONNX: result = _dft(weighted_new_x, ...)
        // 公式: X[k] = sum(x[n] * e^(-i * 2*PI * k * n / N))
        uint phase_idx = phase_acc % fft_length;
        float angle = -2.0 * PI * float(phase_idx) / float(fft_length);

        // 累加 DFT 结果
        sum_r += weighted_new_x * cos(angle);
        sum_i += weighted_new_x * sin(angle);

        phase_acc += k;
    }

    out_result[out_idx]     = sum_r;
    out_result[out_idx + 1] = sum_i;
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"STFTOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"STFTOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        tensor_x = input_tensors[0][0]
        shape_x = input_tensors[0][1]

        hop_length = int(input_tensors[1][0].data()[0])

        tensor_window = None
        shape_window = []
        if len(input_tensors) > 2 and input_tensors[2] is not None:
            tensor_window = input_tensors[2][0]
            shape_window = input_tensors[2][1]

        frame_length_arg = None
        if len(input_tensors) > 3 and input_tensors[3] is not None:
            frame_length_arg = int(input_tensors[3][0].data()[0])

        x_len = shape_x[-1]

        # 逻辑对应 STFT._run 中的处理
        if frame_length_arg is not None:
            window_size = frame_length_arg
        elif tensor_window is not None:
            window_size = shape_window[0]
        else:
            window_size = x_len

        fft_length = window_size

        # 对应 ONNX: n_frames = 1 + (len - n_fft) // hop_length
        # 这里 window_size 充当了 n_fft 的角色
        n_frames = 1 + (x_len - window_size) // hop_length
        if n_frames < 0:
            n_frames = 0

        if self.onesided:
            n_freqs = fft_length // 2 + 1
        else:
            n_freqs = fft_length

        batch_size = int(np.prod(shape_x[:-1])) if len(shape_x) > 1 else 1

        output_shape = shape_x[:-1] + [n_frames, n_freqs, 2]
        total_elements = int(np.prod(output_shape))

        tensor_out = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
        updated_tensors.append(tensor_out)

        if total_elements > 0:
            use_window_val = 1.0 if tensor_window is not None else 0.0
            bind_window_tensor = tensor_window if tensor_window is not None else tensor_x

            params = [
                x_len,
                hop_length,
                window_size,
                fft_length,
                n_frames,
                n_freqs,
                use_window_val
            ]

            workgroup = (n_frames, n_freqs, batch_size)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_x, bind_window_tensor, tensor_out],
                self.compiled_shader,
                workgroup,
                params,
                []
            ))

        return [(tensor_out, output_shape)]

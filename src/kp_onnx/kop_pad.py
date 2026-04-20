import numpy as np
import kp
from .shader_utils import compile_source


class PadOp:
    def __init__(self, manager: kp.Manager, mode='constant'):
        self.manager = manager
        self.mode = mode
        
        # workgroup: (leading, out_len, trailing)
        self.compiled_shader_constant = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  { float in_buf[];  };
layout (binding = 1) writeonly buffer buf_out { float out_buf[]; };

layout (constant_id = 0) const float in_len_f = 0;
layout (constant_id = 1) const float out_len_f = 0;
layout (constant_id = 2) const float trailing_f = 0;
layout (constant_id = 3) const float pad_f = 0;
layout (constant_id = 4) const float fill_f = 0;

void main() {
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_axis_idx = gl_GlobalInvocationID.y;
    uint trailing_idx = gl_GlobalInvocationID.z;
    uint in_len = uint(in_len_f);
    uint out_len = uint(out_len_f);
    uint trailing = uint(trailing_f);
    uint pad = uint(pad_f);
    
    uint out_idx = leading_idx * out_len * trailing + out_axis_idx * trailing + trailing_idx;
    
    if (out_axis_idx >= pad && out_axis_idx < pad + in_len) {
        uint in_axis_idx = out_axis_idx - pad;
        uint in_idx = leading_idx * in_len * trailing + in_axis_idx * trailing + trailing_idx;
        out_buf[out_idx] = in_buf[in_idx];
    } else {
        out_buf[out_idx] = fill_f;
    }
}
""")

        self.compiled_shader_edge = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  { float in_buf[];  };
layout (binding = 1) writeonly buffer buf_out { float out_buf[]; };

layout (constant_id = 0) const float in_len_f = 0;
layout (constant_id = 1) const float out_len_f = 0;
layout (constant_id = 2) const float trailing_f = 0;
layout (constant_id = 3) const float pad_f = 0;

void main() {
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_axis_idx = gl_GlobalInvocationID.y;
    uint trailing_idx = gl_GlobalInvocationID.z;
    uint in_len = uint(in_len_f);
    uint out_len = uint(out_len_f);
    uint trailing = uint(trailing_f);
    uint pad = uint(pad_f);
    
    uint in_axis_idx = clamp(out_axis_idx, pad, pad + in_len - 1) - pad;
    uint out_idx = leading_idx * out_len * trailing + out_axis_idx * trailing + trailing_idx;
    uint in_idx = leading_idx * in_len * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}
""")

        self.compiled_shader_reflect = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  { float in_buf[];  };
layout (binding = 1) writeonly buffer buf_out { float out_buf[]; };

layout (constant_id = 0) const float in_len_f = 0;
layout (constant_id = 1) const float out_len_f = 0;
layout (constant_id = 2) const float trailing_f = 0;
layout (constant_id = 3) const float pad_f = 0;

int reflect(int idx, int len) {
    // 循环反射直到索引在 [0, len-1] 范围内
    int period = 2 * (len - 1);  // 反射周期
    if (period == 0) return 0;   // len = 1 的特殊情况
    idx = ((idx % period) + period) % period;  // 映射到 [0, period-1]
    if (idx >= len) idx = period - idx;        // 后半段反射回来
    return idx;
}

void main() {
    uint leading_idx = gl_GlobalInvocationID.x;
    int out_axis_idx = int(gl_GlobalInvocationID.y);
    uint trailing_idx = gl_GlobalInvocationID.z;
    int in_len = int(in_len_f);
    uint out_len = uint(out_len_f);
    uint trailing = uint(trailing_f);
    int pad = int(pad_f);
    
    uint in_axis_idx = uint(reflect(out_axis_idx - pad, in_len));
    uint out_idx = leading_idx * out_len * trailing + uint(out_axis_idx) * trailing + trailing_idx;
    uint in_idx = leading_idx * uint(in_len) * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}
""")

        self.compiled_shader_wrap = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  { float in_buf[];  };
layout (binding = 1) writeonly buffer buf_out { float out_buf[]; };

layout (constant_id = 0) const float in_len_f = 0;
layout (constant_id = 1) const float out_len_f = 0;
layout (constant_id = 2) const float trailing_f = 0;
layout (constant_id = 3) const float pad_f = 0;

int wrap(int idx, int len) {
    return ((idx % len) + len) % len;
}

void main() {
    uint leading_idx = gl_GlobalInvocationID.x;
    int out_axis_idx = int(gl_GlobalInvocationID.y);
    uint trailing_idx = gl_GlobalInvocationID.z;
    int in_len = int(in_len_f);
    uint out_len = uint(out_len_f);
    uint trailing = uint(trailing_f);
    int pad = int(pad_f);
    
    uint in_axis_idx = uint(wrap(out_axis_idx - pad, in_len));
    uint out_idx = leading_idx * out_len * trailing + uint(out_axis_idx) * trailing + trailing_idx;
    uint in_idx = leading_idx * uint(in_len) * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}
""")

        # Select shader based on mode
        if self.mode == 'constant':
            self.compiled_shader = self.compiled_shader_constant
        elif self.mode == 'edge':
            self.compiled_shader = self.compiled_shader_edge
        elif self.mode == 'reflect':
            self.compiled_shader = self.compiled_shader_reflect
        elif self.mode == 'wrap':
            self.compiled_shader = self.compiled_shader_wrap
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def set_mode(self, mode):
        """Update mode and select the corresponding shader."""
        self.mode = mode
        if mode == 'constant':
            self.compiled_shader = self.compiled_shader_constant
        elif mode == 'edge':
            self.compiled_shader = self.compiled_shader_edge
        elif mode == 'reflect':
            self.compiled_shader = self.compiled_shader_reflect
        elif mode == 'wrap':
            self.compiled_shader = self.compiled_shader_wrap
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __repr__(self):
        return f"PadOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(shape_out)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        pads = input_tensors[1][0].data().astype(int).tolist()
        constant_value = input_tensors[2][0].data()[0] if len(input_tensors) > 2 else 0
        axes = input_tensors[3][0].data().astype(int).tolist() if len(input_tensors) > 3 else None
        
        input_rank = len(shape_in)
        
        # Handle axes parameter
        if axes is None:
            axes = list(range(input_rank))
        else:
            axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
        
        num_axes = len(axes)
        assert num_axes * 2 == len(pads), \
            f"The number of elements in pads should be 2 times the number of axes: {len(pads)} != {num_axes * 2}"
        
        # Build full pad_begin and pad_end arrays
        pad_begin = [0] * input_rank
        pad_end = [0] * input_rank
        for i, axis in enumerate(axes):
            pad_begin[axis] = pads[i]
            pad_end[axis] = pads[num_axes + i]
        
        # 找出需要 padding 的 axes（只处理 pad > 0 的轴）
        axes_to_pad = [axis for axis in range(input_rank) if pad_begin[axis] > 0 or pad_end[axis] > 0]
        
        tensor_out = tensor_in
        current_shape = list(shape_in)
        
        # 逐轴处理，workgroup = (leading, out_len, trailing)
        for axis in axes_to_pad:
            pb, pe = pad_begin[axis], pad_end[axis]
            
            # leading = axis 前面维度的乘积, trailing = axis 后面维度的乘积
            leading = int(np.prod(current_shape[:axis])) if axis > 0 else 1
            in_len = current_shape[axis]
            out_len = in_len + pb + pe
            trailing = int(np.prod(current_shape[axis + 1:])) if axis + 1 < input_rank else 1

            # 更新 shape 并分配新的输出 tensor
            current_shape[axis] = out_len
            tensor_in = tensor_out
            tensor_out = self.manager.tensor(np.zeros(int(np.prod(current_shape)), dtype=np.float32))
            updated_tensors.append(tensor_out)

            # spec_consts: [in_len, out_len, trailing, pad, fill(仅constant)]
            if self.mode == 'constant':
                spec_consts = [in_len, out_len, trailing, pb, constant_value]
            else:
                spec_consts = [in_len, out_len, trailing, pb]

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out],
                self.compiled_shader,
                (leading, out_len, trailing),
                spec_consts,
                []
            ))
        
        return [(tensor_out, current_shape)]

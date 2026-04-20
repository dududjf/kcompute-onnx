import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class PadOp:
    def __init__(self, manager: kp.Manager, mode='constant'):
        self.manager = manager
        self.mode = mode

        # workgroup: (leading, out_len, trailing)
        self.compiled_shader_constant = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint leading_idx  = gl_GlobalInvocationID.x;
    uint out_axis_idx = gl_GlobalInvocationID.y;
    uint trailing_idx = gl_GlobalInvocationID.z;

    uint leading  = params[0];
    uint out_len  = params[1];
    uint trailing = params[2];
    if (leading_idx >= leading || out_axis_idx >= out_len || trailing_idx >= trailing) return;

    uint in_len   = params[3];
    uint pad      = params[4];
    float fill_val = uintBitsToFloat(params[5]);

    uint out_idx = leading_idx * out_len * trailing + out_axis_idx * trailing + trailing_idx;

    if (out_axis_idx >= pad && out_axis_idx < pad + in_len) {{
        uint in_axis_idx = out_axis_idx - pad;
        uint in_idx = leading_idx * in_len * trailing + in_axis_idx * trailing + trailing_idx;
        out_buf[out_idx] = in_buf[in_idx];
    }} else {{
        out_buf[out_idx] = fill_val;
    }}
}}
""")

        self.compiled_shader_edge = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint leading_idx  = gl_GlobalInvocationID.x;
    uint out_axis_idx = gl_GlobalInvocationID.y;
    uint trailing_idx = gl_GlobalInvocationID.z;

    uint leading  = params[0];
    uint out_len  = params[1];
    uint trailing = params[2];
    if (leading_idx >= leading || out_axis_idx >= out_len || trailing_idx >= trailing) return;

    uint in_len   = params[3];
    uint pad      = params[4];

    uint in_axis_idx = clamp(out_axis_idx, pad, pad + in_len - 1) - pad;
    uint out_idx = leading_idx * out_len * trailing + out_axis_idx * trailing + trailing_idx;
    uint in_idx  = leading_idx * in_len * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}}
""")

        self.compiled_shader_reflect = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

int reflect(int idx, int len) {{
    int period = 2 * (len - 1);
    if (period == 0) return 0;
    idx = ((idx % period) + period) % period;
    if (idx >= len) idx = period - idx;
    return idx;
}}

void main() {{
    uint leading_idx  = gl_GlobalInvocationID.x;
    int  out_axis_idx = int(gl_GlobalInvocationID.y);
    uint trailing_idx = gl_GlobalInvocationID.z;

    uint leading  = params[0];
    uint out_len  = params[1];
    uint trailing = params[2];
    if (leading_idx >= leading || uint(out_axis_idx) >= out_len || trailing_idx >= trailing) return;

    int  in_len   = int(params[3]);
    int  pad      = int(params[4]);

    uint in_axis_idx = uint(reflect(out_axis_idx - pad, in_len));
    uint out_idx = leading_idx * out_len * trailing + uint(out_axis_idx) * trailing + trailing_idx;
    uint in_idx  = leading_idx * uint(in_len) * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}}
""")

        self.compiled_shader_wrap = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

int wrap(int idx, int len) {{
    return ((idx % len) + len) % len;
}}

void main() {{
    uint leading_idx  = gl_GlobalInvocationID.x;
    int  out_axis_idx = int(gl_GlobalInvocationID.y);
    uint trailing_idx = gl_GlobalInvocationID.z;

    uint leading  = params[0];
    uint out_len  = params[1];
    uint trailing = params[2];
    if (leading_idx >= leading || uint(out_axis_idx) >= out_len || trailing_idx >= trailing) return;

    int  in_len   = int(params[3]);
    int  pad      = int(params[4]);

    uint in_axis_idx = uint(wrap(out_axis_idx - pad, in_len));
    uint out_idx = leading_idx * out_len * trailing + uint(out_axis_idx) * trailing + trailing_idx;
    uint in_idx  = leading_idx * uint(in_len) * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}}
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

    __str__ = __repr__

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

            # params: [leading, out_len, trailing, in_len, pad, (fill)]
            if self.mode == 'constant':
                fill_uint = np.float32(constant_value).view(np.uint32)
                params = np.array([leading, out_len, trailing, in_len, pb, fill_uint], dtype=np.uint32)
            else:
                params = np.array([leading, out_len, trailing, in_len, pb], dtype=np.uint32)

            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            workgroup = (
                (leading + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (out_len + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                (trailing + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
            )

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out, param_in],
                self.compiled_shader,
                workgroup,
            ))

        return [(tensor_out, current_shape)]

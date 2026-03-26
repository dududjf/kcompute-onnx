import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class CenterCropPadOp:
    def __init__(self, manager: kp.Manager, axes=None):
        self.manager = manager
        self.axes = axes

        # Shader for center-padding (target > input on this axis)
        # workgroup: (leading, out_len, trailing)
        # params: [leading, out_len, trailing, in_len, pad_begin]
        self.compiled_shader_pad = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint leading_idx  = gl_GlobalInvocationID.x;
    uint out_axis_idx = gl_GlobalInvocationID.y;
    uint trailing_idx = gl_GlobalInvocationID.z;

    uint leading   = params[0];
    uint out_len   = params[1];
    uint trailing  = params[2];
    if (leading_idx >= leading || out_axis_idx >= out_len || trailing_idx >= trailing) return;

    uint in_len    = params[3];
    uint pad_begin = params[4];

    uint out_idx = leading_idx * out_len * trailing + out_axis_idx * trailing + trailing_idx;

    if (out_axis_idx >= pad_begin && out_axis_idx < pad_begin + in_len) {{
        uint in_axis_idx = out_axis_idx - pad_begin;
        uint in_idx = leading_idx * in_len * trailing + in_axis_idx * trailing + trailing_idx;
        out_buf[out_idx] = in_buf[in_idx];
    }} else {{
        out_buf[out_idx] = 0.0;
    }}
}}
""")

        # Shader for center-cropping (target < input on this axis)
        # workgroup: (leading, out_len, trailing)
        # params: [leading, out_len, trailing, in_len, crop_begin]
        self.compiled_shader_crop = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint leading_idx  = gl_GlobalInvocationID.x;
    uint out_axis_idx = gl_GlobalInvocationID.y;
    uint trailing_idx = gl_GlobalInvocationID.z;

    uint leading    = params[0];
    uint out_len    = params[1];
    uint trailing   = params[2];
    if (leading_idx >= leading || out_axis_idx >= out_len || trailing_idx >= trailing) return;

    uint in_len     = params[3];
    uint crop_begin = params[4];

    uint in_axis_idx = out_axis_idx + crop_begin;
    uint out_idx = leading_idx * out_len * trailing + out_axis_idx * trailing + trailing_idx;
    uint in_idx  = leading_idx * in_len * trailing + in_axis_idx * trailing + trailing_idx;
    out_buf[out_idx] = in_buf[in_idx];
}}
""")

    def __repr__(self):
        return f"CenterCropPadOp({self.manager.get_device_properties()['device_name']})"

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
        shape_target = input_tensors[1][0].data().astype(int).tolist()

        input_rank = len(shape_in)

        axes = self.axes
        if axes is None:
            axes = list(range(input_rank))
        else:
            axes = [a if a >= 0 else a + input_rank for a in axes]

        tensor_out = tensor_in
        current_shape = list(shape_in)

        # Check if any axis needs processing
        axes_to_process = [(i, axis) for i, axis in enumerate(axes) if shape_target[i] != current_shape[axis]]
        if not axes_to_process:
            # No-op: allocate a copy to avoid returning the input tensor directly
            tensor_out = self.manager.tensor(np.zeros(int(np.prod(current_shape)), dtype=np.float32))
            updated_tensors.append(tensor_out)

            leading = 1
            out_len = int(np.prod(current_shape))
            trailing = 1
            params = np.array([leading, out_len, trailing, out_len, 0], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            workgroup = (
                1,
                (out_len + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                1,
            )
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out, param_in],
                self.compiled_shader_crop,
                workgroup,
            ))
            return [(tensor_out, current_shape)]

        for i, axis in enumerate(axes):
            dim = current_shape[axis]
            sh = shape_target[i]

            if sh == dim:
                continue

            leading = int(np.prod(current_shape[:axis])) if axis > 0 else 1
            trailing = int(np.prod(current_shape[axis + 1:])) if axis + 1 < input_rank else 1

            if sh < dim:
                # Center crop
                d = dim - sh
                crop_begin = d // 2
                out_len = sh

                current_shape[axis] = out_len
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(np.zeros(int(np.prod(current_shape)), dtype=np.float32))
                updated_tensors.append(tensor_out)

                params = np.array([leading, out_len, trailing, dim, crop_begin], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (leading + LOCAL_X_3D - 1) // LOCAL_X_3D,
                    (out_len + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                    (trailing + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_out, param_in],
                    self.compiled_shader_crop,
                    workgroup,
                ))

            else:
                # Center pad
                d = sh - dim
                pad_begin = d // 2
                out_len = sh

                current_shape[axis] = out_len
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(np.zeros(int(np.prod(current_shape)), dtype=np.float32))
                updated_tensors.append(tensor_out)

                params = np.array([leading, out_len, trailing, dim, pad_begin], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (leading + LOCAL_X_3D - 1) // LOCAL_X_3D,
                    (out_len + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                    (trailing + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_out, param_in],
                    self.compiled_shader_pad,
                    workgroup,
                ))

        return [(tensor_out, current_shape)]

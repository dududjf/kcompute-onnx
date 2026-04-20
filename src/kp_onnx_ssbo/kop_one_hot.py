import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class OneHotOp:
    def __init__(self, manager: kp.Manager, axis: int = -1):
        self.manager = manager
        self.axis = axis
        self.compiled_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly buffer buf_indices {{ float indices_buf[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer buf_output {{ float output_buf[]; }};
layout (std430, set = 0, binding = 2) readonly buffer UIParams    {{ uint params[]; }};

void main() {{
    uint left_size = params[0];
    uint depth = params[1];
    uint right_size = params[2];
    float off_value = uintBitsToFloat(params[3]);
    float on_value = uintBitsToFloat(params[4]);

    uint left_idx = gl_GlobalInvocationID.x;
    uint depth_idx = gl_GlobalInvocationID.y;
    uint right_idx = gl_GlobalInvocationID.z;

    if (left_idx >= left_size || depth_idx >= depth || right_idx >= right_size) return;

    uint indices_idx = left_idx * right_size + right_idx;
    uint output_idx = (left_idx * depth + depth_idx) * right_size + right_idx;

    int index = int(indices_buf[indices_idx]);
    uint index_value = uint((index % int(depth) + int(depth)) % int(depth));

    if (depth_idx == index_value) {{
        output_buf[output_idx] = on_value;
    }} else {{
        output_buf[output_idx] = off_value;
    }}
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"OneHotOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        if tensor_out is not None:
            output = tensor_out.data().reshape(output_shape)
        else:
            output = np.array([], dtype=np.float32).reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_indices, shape_indices = input_tensors[0]
        tensor_depth, shape_depth = input_tensors[1]
        tensor_values, shape_values = input_tensors[2]

        axis = self.axis
        if axis < 0:
            axis += len(shape_indices) + 1

        values = tensor_values.data()
        off_value, on_value = values[0], values[1]
        depth_value = int(tensor_depth.data()[0])

        if depth_value == 0:
            ls = shape_indices[0:axis]
            rs = shape_indices[axis:]
            shape_out = ls[:] + [depth_value] + rs[:]
            return [(None, shape_out)]

        else:
            ls = shape_indices[0:axis]
            rs = shape_indices[axis:]
            left_size = int(np.prod(ls)) if len(ls) > 0 else 1
            right_size = int(np.prod(rs)) if len(rs) > 0 else 1

            shape_out = ls[:] + [depth_value] + rs[:]
            total_size = left_size * depth_value * right_size

            tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
            updated_tensors.append(tensor_out)

            off_uint = np.uint32(np.float32(off_value).view(np.uint32))
            on_uint = np.uint32(np.float32(on_value).view(np.uint32))
            params = np.array([left_size, depth_value, right_size, off_uint, on_uint], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            workgroup = ((left_size + LOCAL_X_3D - 1) // LOCAL_X_3D, (depth_value + LOCAL_Y_3D - 1) // LOCAL_Y_3D, (right_size + LOCAL_Z_3D - 1) // LOCAL_Z_3D)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_indices, tensor_out, param_in],
                self.compiled_shader,
                workgroup,
            ))

            return [(tensor_out, shape_out)]

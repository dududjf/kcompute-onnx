import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D

DEFAULT_AXIS = 0


class ArgMinOp:

    def __init__(self, manager: kp.Manager, axis=DEFAULT_AXIS, keepdims=True, select_last_index=False):
        self.axis = axis
        self.keepdims = keepdims
        self.select_last_index = select_last_index
        self.manager = manager
        self.compiled_shader = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_buf[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ int out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{
    uint bound_x;
    uint bound_y;
    uint axis_size;
    uint stride_after;
    uint select_last;
}};

void main() {{
    uint out_before = gl_GlobalInvocationID.x;
    uint out_after  = gl_GlobalInvocationID.y;
    
    if (out_before >= bound_x || out_after >= bound_y) return;

    uint base = (out_before * axis_size) * stride_after + out_after;

    float min_val = in_buf[base];
    uint  min_idx = 0u;
    base += stride_after;

    if (select_last == 1u) {{
        for (uint i = 1u; i < axis_size; ++i, base += stride_after) {{
            float v = in_buf[base];
            if (v <= min_val) {{
                min_val = v;
                min_idx = i;
            }}
        }}
    }} 
    else {{
        for (uint i = 1u; i < axis_size; ++i, base += stride_after) {{
            float v = in_buf[base];
            if (v < min_val) {{
                min_val = v;
                min_idx = i;
            }}
        }}
    }}
    out_buf[out_before * stride_after + out_after] = int(min_idx);
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ArgMinOp({dev})"

    __str__ = __repr__

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
        tensor_in, shape = input_tensors[0]
        rank = len(shape)

        axis = self.axis if self.axis >= 0 else rank + self.axis
        assert 0 <= axis < rank, "axis out of range"

        axis_size = int(shape[axis])
        stride_before = int(np.prod(shape[:axis])) if axis > 0 else 1
        stride_after  = int(np.prod(shape[axis+1:])) if axis < rank - 1 else 1

        if self.keepdims:
            out_shape = [1 if i == axis else d for i, d in enumerate(shape)]
        else:
            out_shape = [d for i, d in enumerate(shape) if i != axis]

        output_size = stride_before * stride_after

        tensor_out = self.manager.tensor_t(np.zeros(output_size, dtype=np.int32), tensor_type=kp.TensorTypes.device)
        updated_tensors.append(tensor_out)

        # 创建参数张量并立即同步到GPU
        params = np.array([
            stride_before,                      # bound_x
            stride_after,                       # bound_y
            axis_size,
            stride_after,
            1 if self.select_last_index else 0  # select_last
        ], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        # 计算工作组数量
        group_x = (stride_before + LOCAL_X_2D - 1) // LOCAL_X_2D
        group_y = (stride_after + LOCAL_Y_2D - 1) // LOCAL_Y_2D
        workgroup = (group_x, group_y, 1)

        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out, param_in],
                self.compiled_shader,
                workgroup
            )
        )

        return [(tensor_out, out_shape)]


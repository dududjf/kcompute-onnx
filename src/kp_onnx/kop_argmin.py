import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_AXIS = 0


class ArgMinOp:

    def __init__(self, manager: kp.Manager, axis=DEFAULT_AXIS, keepdims=True, select_last_index=False):
        self.axis = axis
        self.keepdims = keepdims
        self.select_last_index = select_last_index
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(binding=0) readonly buffer InBuf  { float in_buf[]; };
layout(binding=1) writeonly buffer OutBuf{ int   out_buf[]; };

layout(constant_id=0) const float axis_size_f    = 0.0;
layout(constant_id=1) const float stride_after_f = 0.0;
layout(constant_id=2) const float select_last_f  = 0.0;

void main() {
    uint axis_size    = uint(axis_size_f);
    uint stride_after = uint(stride_after_f);
    uint select_last  = uint(select_last_f);

    uint out_before = gl_GlobalInvocationID.x;
    uint out_after  = gl_GlobalInvocationID.y;

    uint base = (out_before * axis_size) * stride_after + out_after;

    float min_val = in_buf[base];
    uint  min_idx = 0u;
    base += stride_after;

    if (select_last == 1u) {
        for (uint i = 1u; i < axis_size; ++i, base += stride_after) {
            float v = in_buf[base];
            if (v <= min_val) {
                min_val = v;
                min_idx = i;
            }
        }
    } 
    else {
        for (uint i = 1u; i < axis_size; ++i, base += stride_after) {
            float v = in_buf[base];
            if (v < min_val) {
                min_val = v;
                min_idx = i;
            }
        }
    }
    out_buf[out_before * stride_after + out_after] = int(min_idx);
}
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
        if output_size <= 0:
            output_size = 1

        tensor_out = self.manager.tensor_t(np.zeros(output_size, dtype=np.int32), tensor_type=kp.TensorTypes.device)
        updated_tensors.append(tensor_out)
        workgroup = (stride_before, stride_after, 1)
        shader = self.shader
        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                shader,
                workgroup,
                [axis_size, stride_after, self.select_last_index],
                []
            )
        )

        return [(tensor_out, out_shape)]

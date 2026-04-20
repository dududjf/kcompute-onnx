import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class EyeLikeOp:

    def __init__(self, manager: kp.Manager, k=0):
        self.k = k
        self.manager = manager
        self.compiled_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout(std430, set = 0, binding = 0) writeonly buffer OutBuf {{ float out_buf[]; }};
layout(std430, set = 0, binding = 1) readonly  buffer UIParams {{
    uint cols;
    uint rows;
    int  k;
}};

void main() {{
    uint c = gl_GlobalInvocationID.x;
    uint r = gl_GlobalInvocationID.y;

    if (c >= cols || r >= rows) return;

    out_buf[r * cols + c] = ((int(c) - int(r)) == k) ? 1.0 : 0.0;
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"EyeLikeOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        if input_tensors:
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
        shape_in = input_tensors[0][1]
        assert 1 <= len(shape_in) <= 2, f"EyeLike only accepts 1D or 2D input, got {shape_in}"

        if len(shape_in) == 1:
            rows = cols = shape_in[0]
        else:
            rows, cols = shape_in
        out_shape = [rows, cols]

        size = rows * cols
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params_data = np.array([cols, rows, self.k], dtype=np.uint32)
        param_in = self.manager.tensor_t(params_data, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        updated_tensors.append(param_in)

        workgroup = ((cols + LOCAL_X_2D - 1) // LOCAL_X_2D, (rows + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_out, param_in],
            self.compiled_shader,
            workgroup
        ))

        return [(tensor_out, out_shape)]


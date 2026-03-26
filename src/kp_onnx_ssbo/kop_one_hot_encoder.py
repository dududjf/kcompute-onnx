import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class OneHotEncoderOp:
    def __init__(self, manager: kp.Manager, cats_int64s=None, cats_strings=None, zeros=1):
        self.manager = manager
        self.cats_int64s = cats_int64s
        self.cats_strings = cats_strings
        self.zeros = zeros
        self.compiled_shader = compile_source(f"""
#version 450
layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf        {{ int input_data[]; }};
layout(std430, set = 0, binding = 1) readonly  buffer CategoriesBuf {{ int categories[]; }};
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf       {{ float output_data[]; }};
layout(std430, set = 0, binding = 3) readonly  buffer UIParams     {{ uint params[]; }};

void main() {{
    uint input_idx = gl_GlobalInvocationID.x;
    uint category_idx = gl_GlobalInvocationID.y;
    uint input_size = params[0];
    uint num_categories = params[1];
    if (input_idx >= input_size || category_idx >= num_categories) return;

    uint output_idx = input_idx * num_categories + category_idx;
    int input_value = input_data[input_idx];
    int category_value = categories[category_idx];
    output_data[output_idx] = input_value == category_value ? 1.0 : 0.0;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"OneHotEncoderOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.int32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.int32)
            tensor = self.manager.tensor_t(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        tensors_to_sync = [t[0] for t in input_tensors] + updated_tensors[:-1]
        seq.record(kp.OpTensorSyncDevice(tensors_to_sync))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([t[0] for t in output_tensor_and_shape]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]

        assert len(shape_in) <= 2, f"This operator is not implemented for shape {shape_in}."
        assert self.cats_int64s is not None or self.cats_strings is not None, "No encoding was defined."

        if self.cats_int64s is not None:
            categories = np.array(self.cats_int64s, dtype=np.int32)
        else:
            assert False, "String encoding is not implemented yet."

        num_categories = len(categories)
        input_size = int(np.prod(shape_in))
        
        shape_out = shape_in[:] + [num_categories]
        total_output_size = input_size * num_categories
        
        tensor_categories = self.manager.tensor_t(categories)
        updated_tensors.append(tensor_categories)
        
        tensor_out = self.manager.tensor(np.zeros(total_output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params = np.array([input_size, num_categories], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_categories, tensor_out, param_in],
            self.compiled_shader,
            (input_size, num_categories, 1),
        ))
        
        return [(tensor_out, shape_out)]

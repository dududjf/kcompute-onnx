import numpy as np
import kp
from .shader_utils import compile_source


class LabelEncoderOp:
    def __init__(self, manager: kp.Manager, 
                 default_float=-0.0, 
                 default_int64=-1,
                 default_string='_Unused',
                 default_tensor=None,
                 keys_floats=None, 
                 keys_int64s=None,
                 keys_strings=None,
                 keys_tensor=None,
                 values_floats=None, 
                 values_int64s=None,
                 values_strings=None,
                 values_tensor=None):
        self.manager = manager
        self.default_float = default_float
        self.default_int64 = default_int64
        self.default_string = default_string
        self.default_tensor = default_tensor
        
        # Preprocess keys and values: sort and deduplicate (keep last value for duplicate keys)
        # Priority: floats > int64s > strings > tensor (consistent with ONNX spec)
        self.sorted_keys = None
        self.sorted_values = None
        self.dtype = None
        self.default_value = None
        
        self._preprocess_keys_values(keys_floats, keys_int64s, keys_strings, keys_tensor,
                                     values_floats, values_int64s, values_strings, values_tensor,
                                     default_float, default_int64, default_tensor)
        
        # Float shader with binary search
        self.compiled_shader_float = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_input   { float input_buf[];  };
layout (binding = 1) readonly  buffer buf_keys    { float keys_buf[];   };
layout (binding = 2) readonly  buffer buf_values  { float values_buf[]; };
layout (binding = 3) writeonly buffer buf_output  { float output_buf[]; };

layout (constant_id = 0) const float num_mappings_f = 0;
layout (constant_id = 1) const float default_value_f = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float input_val = input_buf[idx];
    int num_mappings = int(num_mappings_f);
    
    // Binary search
    int left = 0;
    int right = num_mappings - 1;
    float output_val = default_value_f;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        float key = keys_buf[mid];
        if (key == input_val) {
            output_val = values_buf[mid];
            break;
        } else if (key < input_val) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    output_buf[idx] = output_val;
}
""")

        # Int shader with binary search
        self.compiled_shader_int = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_input   { int input_buf[];  };
layout (binding = 1) readonly  buffer buf_keys    { int keys_buf[];   };
layout (binding = 2) readonly  buffer buf_values  { int values_buf[]; };
layout (binding = 3) writeonly buffer buf_output  { int output_buf[]; };

layout (constant_id = 0) const float num_mappings_f = 0;
layout (constant_id = 1) const float default_value_f = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    int input_val = input_buf[idx];
    int num_mappings = int(num_mappings_f);
    int default_value = int(default_value_f);
    
    // Binary search
    int left = 0;
    int right = num_mappings - 1;
    int output_val = default_value;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        int key = keys_buf[mid];
        if (key == input_val) {
            output_val = values_buf[mid];
            break;
        } else if (key < input_val) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    output_buf[idx] = output_val;
}
""")

    def _preprocess_keys_values(self, keys_floats, keys_int64s, keys_strings, keys_tensor,
                                values_floats, values_int64s, values_strings, values_tensor,
                                default_float, default_int64, default_tensor):
        """Preprocess keys and values: sort and deduplicate (keep last value for duplicate keys)."""
        # Priority: floats > int64s > strings > tensor (consistent with ONNX spec)
        if keys_floats is not None and len(keys_floats) > 0:
            keys = np.array(keys_floats, dtype=np.float32)
            values = np.array(values_floats, dtype=np.float32)
            self.default_value = default_float
            self.dtype = np.float32
        elif keys_int64s is not None and len(keys_int64s) > 0:
            keys = np.array(keys_int64s, dtype=np.int32)
            values = np.array(values_int64s, dtype=np.int32)
            self.default_value = default_int64
            self.dtype = np.int32
        elif keys_strings is not None and len(keys_strings) > 0:
            raise ValueError("String keys are not supported in GPU implementation yet.")
        elif keys_tensor is not None and values_tensor is not None:
            keys = keys_tensor.astype(np.float32)
            values = values_tensor.astype(np.float32)
            self.default_value = float(default_tensor) if default_tensor is not None else 0.0
            self.dtype = np.float32
        else:
            raise ValueError("Keys must be provided.")
        
        # Sort and deduplicate (keep last value for duplicate keys)
        key_value_map = {}
        for k, v in zip(keys, values):
            key_value_map[k] = v
        self.sorted_keys = np.array(sorted(key_value_map.keys()), dtype=self.dtype)
        self.sorted_values = np.array([key_value_map[k] for k in self.sorted_keys], dtype=self.dtype)

    def set_keys_values(self, keys_floats=None, keys_int64s=None, keys_strings=None, keys_tensor=None,
                        values_floats=None, values_int64s=None, values_strings=None, values_tensor=None,
                        default_float=None, default_int64=None, default_tensor=None):
        """Update keys and values attributes and reprocess."""
        if default_float is not None:
            self.default_float = default_float
        if default_int64 is not None:
            self.default_int64 = default_int64
        if default_tensor is not None:
            self.default_tensor = default_tensor
        
        self._preprocess_keys_values(keys_floats, keys_int64s, keys_strings, keys_tensor,
                                     values_floats, values_int64s, values_strings, values_tensor,
                                     self.default_float, self.default_int64, self.default_tensor)

    def __repr__(self):
        return f"LabelEncoderOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(self.dtype)
            if self.dtype == np.int32:
                tensor = self.manager.tensor_t(numpy_in)
            else:
                tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]] + updated_tensors))
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
        
        assert len(self.sorted_keys) == len(self.sorted_values), \
            f"Keys and values must have the same length: {len(self.sorted_keys)} != {len(self.sorted_values)}"
        
        num_mappings = len(self.sorted_keys)
        total_size = int(np.prod(shape_in))
        
        # Create tensors for keys and values
        if self.dtype == np.int32:
            tensor_keys = self.manager.tensor_t(self.sorted_keys)
            tensor_values = self.manager.tensor_t(self.sorted_values)
            tensor_out = self.manager.tensor_t(np.zeros(total_size, dtype=np.int32))
            compiled_shader = self.compiled_shader_int
        else:
            tensor_keys = self.manager.tensor(self.sorted_keys)
            tensor_values = self.manager.tensor(self.sorted_values)
            tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
            compiled_shader = self.compiled_shader_float
        
        updated_tensors.append(tensor_keys)
        updated_tensors.append(tensor_values)
        updated_tensors.append(tensor_out)

        # Create algorithm
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_keys, tensor_values, tensor_out],
            compiled_shader,
            (total_size, 1, 1),
            [num_mappings, self.default_value],
            []
        ))
        
        return [(tensor_out, shape_in)]


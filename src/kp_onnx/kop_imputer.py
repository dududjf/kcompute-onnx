import numpy as np
import kp
from .shader_utils import compile_source


class ImputerOp:
    def __init__(self, manager: kp.Manager, imputed_value_floats=None, imputed_value_int64s=None,
                 replaced_value_float=0.0, replaced_value_int64=0):
        self.manager = manager
        self.imputed_value_floats = imputed_value_floats
        self.imputed_value_int64s = imputed_value_int64s
        self.replaced_value_float = replaced_value_float
        self.replaced_value_int64 = replaced_value_int64
        # Float shader
        self.compiled_shader_float = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_input    { float input_buf[];    };
layout (binding = 1) readonly  buffer buf_imputed  { float imputed_buf[];  };
layout (binding = 2) readonly  buffer buf_replaced { float replaced_buf[]; };
layout (binding = 3) writeonly buffer buf_output   { float output_buf[];   };

layout (constant_id = 0) const float n_cols_f = 0;

bool should_replace(float val, float replaced_value) {
    if (isnan(val) && isnan(replaced_value)) {
        return true;
    }
    if (!isnan(val) && !isnan(replaced_value) && val == replaced_value) {
        return true;
    }
    return false;
}

void main() {
    uint row_idx = gl_GlobalInvocationID.x;
    uint col_idx = gl_GlobalInvocationID.y;
    
    uint n_cols = uint(n_cols_f);
    float replaced_value = replaced_buf[0];
    
    uint idx = row_idx * n_cols + col_idx;
    float input_val = input_buf[idx];
    
    if (should_replace(input_val, replaced_value)) {
        output_buf[idx] = imputed_buf[col_idx];
    } else {
        output_buf[idx] = input_val;
    }
}
""")

        # Int shader
        self.compiled_shader_int = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_input    { int input_buf[];    };
layout (binding = 1) readonly  buffer buf_imputed  { int imputed_buf[];  };
layout (binding = 2) readonly  buffer buf_replaced { int replaced_buf[]; };
layout (binding = 3) writeonly buffer buf_output   { int output_buf[];   };

layout (constant_id = 0) const float n_cols_f = 0;

void main() {
    uint row_idx = gl_GlobalInvocationID.x;
    uint col_idx = gl_GlobalInvocationID.y;
    
    uint n_cols = uint(n_cols_f);
    int replaced_value = replaced_buf[0];
    
    uint idx = row_idx * n_cols + col_idx;
    int input_val = input_buf[idx];
    
    if (input_val == replaced_value) {
        output_buf[idx] = imputed_buf[col_idx];
    } else {
        output_buf[idx] = input_val;
    }
}
""")

    def __repr__(self):
        return f"ImputerOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def set_imputed_values(self, imputed_value_floats=None, imputed_value_int64s=None,
                           replaced_value_float=None, replaced_value_int64=None):
        """动态更新 Imputer 的属性，避免重新创建实例和重新编译 shader"""
        if imputed_value_floats is not None:
            self.imputed_value_floats = imputed_value_floats
        if imputed_value_int64s is not None:
            self.imputed_value_int64s = imputed_value_int64s
        if replaced_value_float is not None:
            self.replaced_value_float = replaced_value_float
        if replaced_value_int64 is not None:
            self.replaced_value_int64 = replaced_value_int64

    def run(self, *inputs):
        # 根据属性判断输入数据类型（优先级：float > int64）
        if self.imputed_value_int64s is not None and len(self.imputed_value_int64s) > 0:
            dtype = np.int32
        else:
            dtype = np.float32
        
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(dtype)
            if dtype == np.int32:
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

        if self.imputed_value_floats is not None and len(self.imputed_value_floats) > 0:
            imputed_source = self.imputed_value_floats
            replaced_value = self.replaced_value_float
            dtype = np.float32
        elif self.imputed_value_int64s is not None and len(self.imputed_value_int64s) > 0:
            imputed_source = self.imputed_value_int64s
            replaced_value = self.replaced_value_int64
            dtype = np.int32
        else:
            raise "Imputed values must be provided."

        if isinstance(imputed_source, list):
            imputed_source = np.array(imputed_source, dtype=dtype)
        assert len(shape_in) == 2, f"x must be a matrix but shape is {shape_in}"
        assert imputed_source.shape[0] in [1, shape_in[1]], \
            f"Dimension mismatch {imputed_source.shape[0]} != {shape_in[1]}"

        n_rows = shape_in[0]
        n_cols = shape_in[1]

        # 扩展 imputed_source 到 n_cols 大小（广播单值或直接使用）
        if len(imputed_source) == 1:
            imputed_expanded = imputed_source[[0] * n_cols]
        else:
            imputed_expanded = imputed_source.astype(dtype)
        
        if dtype == np.int32:
            tensor_imputed = self.manager.tensor_t(imputed_expanded)
            updated_tensors.append(tensor_imputed)

            tensor_replaced = self.manager.tensor_t(np.array([replaced_value], dtype=np.int32))
            updated_tensors.append(tensor_replaced)

            tensor_out = self.manager.tensor_t(np.zeros(n_rows * n_cols, dtype=np.int32))
            updated_tensors.append(tensor_out)

            compiled_shader = self.compiled_shader_int
        else:
            tensor_imputed = self.manager.tensor(imputed_expanded)
            updated_tensors.append(tensor_imputed)

            tensor_replaced = self.manager.tensor(np.array([replaced_value], dtype=np.float32))
            updated_tensors.append(tensor_replaced)

            tensor_out = self.manager.tensor(np.zeros(n_rows * n_cols, dtype=np.float32))
            updated_tensors.append(tensor_out)

            compiled_shader = self.compiled_shader_float

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_imputed, tensor_replaced, tensor_out],
            compiled_shader,
            (n_rows, n_cols, 1),
            [n_cols],
            []
        ))
        
        return [(tensor_out, shape_in)]


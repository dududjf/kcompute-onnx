import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class ImputerOp:
    def __init__(self, manager: kp.Manager, imputed_value_floats=None, imputed_value_int64s=None,
                 replaced_value_float=0.0, replaced_value_int64=0):
        self.manager = manager
        self.imputed_value_floats = imputed_value_floats
        self.imputed_value_int64s = imputed_value_int64s
        self.replaced_value_float = replaced_value_float
        self.replaced_value_int64 = replaced_value_int64
        # Float shader
        self.compiled_shader_float = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout(std430, set = 0, binding = 0) readonly  buffer InBuf     {{ float in_tensor[];      }};
layout(std430, set = 0, binding = 1) readonly  buffer ImputedBuf {{ float imputed_tensor[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer ReplacedBuf {{ float replaced_tensor[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf    {{ float out_tensor[];     }};
layout(std430, set = 0, binding = 4) readonly  buffer UIParams  {{ uint params[]; }};

bool should_replace(float val, float replaced_value) {{
    if (isnan(val) && isnan(replaced_value)) {{
        return true;
    }}
    if (!isnan(val) && !isnan(replaced_value) && val == replaced_value) {{
        return true;
    }}
    return false;
}}

void main() 
{{
    uint n_rows = params[0];
    uint n_cols = params[1];
    
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    if(gx >= n_rows || gy >= n_cols) return;
    
    float replaced_value = replaced_tensor[0];
    
    uint idx = gx * n_cols + gy;
    float input_val = in_tensor[idx];
    
    if (should_replace(input_val, replaced_value)) {{
        out_tensor[idx] = imputed_tensor[gy];
    }} else {{
        out_tensor[idx] = input_val;
    }}
}}
""")

        # Int shader
        self.compiled_shader_int = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout(std430, set = 0, binding = 0) readonly  buffer InBuf     {{ int in_tensor[];      }};
layout(std430, set = 0, binding = 1) readonly  buffer ImputedBuf {{ int imputed_tensor[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer ReplacedBuf {{ int replaced_tensor[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf    {{ int out_tensor[];     }};
layout(std430, set = 0, binding = 4) readonly  buffer UIParams  {{ uint params[]; }};

void main() 
{{
    uint n_rows = params[0];
    uint n_cols = params[1];
    
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    if(gx >= n_rows || gy >= n_cols) return;
    
    int replaced_value = replaced_tensor[0];
    
    uint idx = gx * n_cols + gy;
    int input_val = in_tensor[idx];
    
    if (input_val == replaced_value) {{
        out_tensor[idx] = imputed_tensor[gy];
    }} else {{
        out_tensor[idx] = input_val;
    }}
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ImputerOp({device_name})"

    __str__ = __repr__

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

        workgroup = ((n_rows + LOCAL_X_2D - 1) // LOCAL_X_2D, (n_cols + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)
        params = np.array([n_rows, n_cols], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_imputed, tensor_replaced, tensor_out, param_in],
            compiled_shader,
            workgroup,
        ))
        
        return [(tensor_out, shape_in)]

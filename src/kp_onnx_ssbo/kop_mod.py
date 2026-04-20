import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class ModOp:
    def __init__(self, manager: kp.Manager, fmod=False):
        self.fmod = fmod
        self.manager = manager
        # Shader for float32
        self.shader_trunc_f32 = compile_source(f'''
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_out_tensor  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

float nan_to_zero(float x) {{
     return (isnan(x) || isinf(x)) ? 0.0 : x;
}}

void main() {{
    uint size_x_1 = params[0];
    uint size_y_1 = params[1];
    uint size_z_1 = params[2];
    uint size_x_2 = params[3];
    uint size_y_2 = params[4];
    uint size_z_2 = params[5];

    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    if (gx >= max(size_x_1, size_x_2) || gy >= max(size_y_1, size_y_2) || gz >= max(size_z_1, size_z_2)) return;

    uint stride_y_1 = size_z_1;
    uint stride_x_1 = size_y_1 * stride_y_1;
    uint stride_y_2 = size_z_2;
    uint stride_x_2 = size_y_2 * stride_y_2;
    uint stride_y   = max(size_z_1, size_z_2);
    uint stride_x   = max(size_y_1, size_y_2) * stride_y;

    uint x_1 = min(gx, size_x_1 - 1), y_1 = min(gy, size_y_1 - 1), z_1 = min(gz, size_z_1 - 1);
    uint x_2 = min(gx, size_x_2 - 1), y_2 = min(gy, size_y_2 - 1), z_2 = min(gz, size_z_2 - 1);

    uint p_1 = x_1 * stride_x_1 + y_1 * stride_y_1 + z_1;
    uint p_2 = x_2 * stride_x_2 + y_2 * stride_y_2 + z_2;

    float a = in_tensor_1[p_1];
    float b = in_tensor_2[p_2];

    float q = trunc(a / b);
    float r = a - b * q;
    r = nan_to_zero(r);
    out_tensor[gx * stride_x + gy * stride_y + gz] = r;
}}
''')

        # Shader for int32
        self.shader_trunc_i32 = compile_source(f'''
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer buf_in_tensor_1 {{ int in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer buf_in_tensor_2 {{ int in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_out_tensor  {{ int out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint size_x_1 = params[0];
    uint size_y_1 = params[1];
    uint size_z_1 = params[2];
    uint size_x_2 = params[3];
    uint size_y_2 = params[4];
    uint size_z_2 = params[5];

    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    if (gx >= max(size_x_1, size_x_2) || gy >= max(size_y_1, size_y_2) || gz >= max(size_z_1, size_z_2)) return;

    uint stride_y_1 = size_z_1;
    uint stride_x_1 = size_y_1 * stride_y_1;
    uint stride_y_2 = size_z_2;
    uint stride_x_2 = size_y_2 * stride_y_2;
    uint stride_y   = max(size_z_1, size_z_2);
    uint stride_x   = max(size_y_1, size_y_2) * stride_y;

    uint x_1 = min(gx, size_x_1 - 1), y_1 = min(gy, size_y_1 - 1), z_1 = min(gz, size_z_1 - 1);
    uint x_2 = min(gx, size_x_2 - 1), y_2 = min(gy, size_y_2 - 1), z_2 = min(gz, size_z_2 - 1);

    uint p_1 = x_1 * stride_x_1 + y_1 * stride_y_1 + z_1;
    uint p_2 = x_2 * stride_x_2 + y_2 * stride_y_2 + z_2;

    float a = float(in_tensor_1[p_1]);
    float b = float(in_tensor_2[p_2]);

    float q = trunc(a / b);
    float r = a - b * q;
    out_tensor[gx * stride_x + gy * stride_y + gz] = int(r);
}}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ModOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        assert len(inputs) == 2, "ModOp requires 2 inputs"

        input_tensors = []
        for inp in inputs:
            if inp.dtype in (np.float16, np.float32, np.float64):
                numpy_in = inp.reshape(-1).astype(np.float32)
                tensor = self.manager.tensor(numpy_in)
            elif inp.dtype in (np.int32, np.int64):
                numpy_in = inp.reshape(-1).astype(np.int32)
                tensor = self.manager.tensor_t(numpy_in, tensor_type=kp.TensorTypes.device)
            else:
                raise TypeError(f"Unsupported dtype {inp.dtype}. Only float32 and int32 supported.")
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
        assert len(input_tensors) == 2, "ModOp requires 2 inputs"

        input_1 = input_tensors[0][0]
        input_2 = input_tensors[1][0]
        dtype_1 = input_1.data().dtype
        dtype_2 = input_2.data().dtype
        shape_1 = input_tensors[0][1]
        shape_2 = input_tensors[1][1]
        if len(shape_1) < len(shape_2):
            new_shape_1 = [1] * (len(shape_2) - len(shape_1)) + shape_1
            new_shape_2 = shape_2
        elif len(shape_2) < len(shape_1):
            new_shape_1 = shape_1
            new_shape_2 = [1] * (len(shape_1) - len(shape_2)) + shape_2
        else:
            new_shape_1 = shape_1
            new_shape_2 = shape_2
        output_shape = []
        for i in range(len(new_shape_1)):
            if new_shape_1[i] == 1:
                output_shape.append(new_shape_2[i])
            elif new_shape_2[i] == 1:
                output_shape.append(new_shape_1[i])
            else:
                assert new_shape_1[i] == new_shape_2[i], \
                    "ModOp requires each dimension to be one or equal to the corresponding dimension"
                output_shape.append(new_shape_1[i])

        new_in_1 = input_1
        if output_shape[:-2] != new_shape_1[:-2] and not all(e == 1 for e in new_shape_1[:-2]):
            final_shape_1 = output_shape[:-2] + list(new_shape_1[-2:])
            new_in_1 = broadcast_to(input_1, new_shape_1, final_shape_1,
                                    updated_algorithms, updated_tensors, self.manager)
            new_shape_1 = final_shape_1

        new_in_2 = input_2
        if output_shape[:-2] != new_shape_2[:-2] and not all(e == 1 for e in new_shape_2[:-2]):
            final_shape_2 = output_shape[:-2] + list(new_shape_2[-2:])
            new_in_2 = broadcast_to(input_2, new_shape_2, final_shape_2,
                                    updated_algorithms, updated_tensors, self.manager)
            new_shape_2 = final_shape_2

        if len(new_shape_1) == 1:
            size_x_1 = new_shape_1[0]
            size_y_1 = 1
            size_z_1 = 1
            size_x_2 = new_shape_2[0]
            size_y_2 = 1
            size_z_2 = 1
        elif len(new_shape_1) == 2:
            size_x_1 = new_shape_1[0]
            size_y_1 = new_shape_1[1]
            size_z_1 = 1
            size_x_2 = new_shape_2[0]
            size_y_2 = new_shape_2[1]
            size_z_2 = 1
        else:
            size_x_1 = int(np.prod(new_shape_1[:-2]))
            size_y_1 = new_shape_1[-2]
            size_z_1 = new_shape_1[-1]
            size_x_2 = int(np.prod(new_shape_2[:-2]))
            size_y_2 = new_shape_2[-2]
            size_z_2 = new_shape_2[-1]

        size = int(np.prod(output_shape))
        if dtype_1 == np.int32 and dtype_2 == np.int32:
            shader = self.shader_trunc_i32
            tensor_out = self.manager.tensor_t(np.zeros(size, dtype=np.int32), tensor_type=kp.TensorTypes.device)
            updated_tensors.append(tensor_out)
        else:
            shader = self.shader_trunc_f32
            tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_tensors.append(tensor_out)

        params = [size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2]
        param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        updated_tensors.append(param_in)

        max_x = max(size_x_1, size_x_2)
        max_y = max(size_y_1, size_y_2)
        max_z = max(size_z_1, size_z_2)

        workgroup = (
            (max_x + LOCAL_X_3D - 1) // LOCAL_X_3D,
            (max_y + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
            (max_z + LOCAL_Z_3D - 1) // LOCAL_Z_3D
        )

        updated_algorithms.append(
            self.manager.algorithm(
                [new_in_1, new_in_2, tensor_out, param_in],
                shader,
                workgroup
            )
        )

        return [(tensor_out, output_shape)]


import subprocess
import tempfile
import os
import numpy as np
import kp

LOCAL_X_1D = 256
LOCAL_X_2D = 16
LOCAL_Y_2D = 16
LOCAL_X_3D = 16
LOCAL_Y_3D = 8
LOCAL_Z_3D = 8


def compile_source(glsl_code):
    """
    Compile GLSL to SpirV and return as bytes.
    """

    if not isinstance(glsl_code, str):
        raise TypeError("glslangValidator expects a string.")

    filename1 = os.path.join(tempfile.gettempdir(), "x.txt")
    filename2 = os.path.join(tempfile.gettempdir(), "x.spv")

    with open(filename1, "wb") as f:
        f.write(glsl_code.encode())

    # Note: -O means optimize, use -O0 to disable optimization
    try:
        stdout = subprocess.check_output(
            ["glslangValidator", "-S", "comp", "-V", "-Os", "-o", filename2, filename1], stderr=subprocess.STDOUT
        )
        stdout  # noqa - not used
    except subprocess.CalledProcessError as err:
        e = "Could not compile glsl to Spir-V:\n" + err.output.decode()
        raise Exception(e)

    with open(filename2, "rb") as f:
        binary = f.read()

    return binary


_broadcast_code = compile_source(f'''
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main()
{{
    uint block_1 = params[0], block_2 = params[1], bound_x = params[2], bound_y = params[3];
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    if (gx >= bound_x || gy >= bound_y) return;
    uint in_offset = gx * block_1;
    uint out_offset = gx * block_2 + gy * block_1;
    for(uint gz = 0; gz < block_1; gz++, in_offset++, out_offset++)
        out_tensor[out_offset] = in_tensor[in_offset];
}}''')


def broadcast_to(tensor_in, org_shape, new_shape, out_algorithms, out_next_tensors, manager: kp.Manager):
    """
    Broadcasts a tensor to a new shape.
    """
    tensor_out = tensor_in
    block_1 = 1
    end = len(org_shape) - 1
    while end >= 0:
        start = end
        if org_shape[start] == 1 and new_shape[start] > 1:
            while start >= 0 and org_shape[start] == 1:
                start -= 1
        if start < end:
            tensor_in = tensor_out
            group_x = np.prod(org_shape[:start+1]) if start >= 0 else 1
            group_y = np.prod(new_shape[start+1:end+1])
            block_2 = block_1 * group_y
            workgroup = ((group_x + LOCAL_X_2D - 1) // LOCAL_X_2D, (group_y + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)
            params = [block_1, block_2, group_x, group_y]
            param_in = manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
            manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            tensor_out = manager.tensor(np.zeros(group_x * block_2, dtype=np.float32))
            out_algorithms.append(manager.algorithm([tensor_in, tensor_out, param_in],
                                                    _broadcast_code,
                                                    workgroup))
            out_next_tensors.append(tensor_out)
            block_1 = block_2
            end = start
        else:
            block_1 *= new_shape[end]
            end -= 1
    return tensor_out

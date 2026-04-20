import subprocess
import tempfile
import os
import numpy as np
import kp


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


_broadcast_code = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) buffer buf_in_tensor { float in_tensor[]; };
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; };
layout (constant_id = 0) const float block_1f = 0;
layout (constant_id = 1) const float block_2f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint block_1 = uint(block_1f);
    uint block_2 = uint(block_2f);
    uint in_offset = gx * block_1;
    uint out_offset = gx * block_2 + gy * block_1;
    for(uint gz = 0; gz < block_1; gz++, in_offset++, out_offset++)
        out_tensor[out_offset] = in_tensor[in_offset];
}''')


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
            workgroup = (group_x, group_y, 1)
            np_array = np.zeros(group_x * block_2, dtype=np.float32)
            tensor_out = manager.tensor(np_array)
            out_algorithms.append(manager.algorithm([tensor_in, tensor_out],
                                                    _broadcast_code,
                                                    workgroup,
                                                    [block_1, block_2],
                                                    []))
            out_next_tensors.append(tensor_out)
            block_1 = block_2
            end = start
        else:
            block_1 *= new_shape[end]
            end -= 1
    return tensor_out

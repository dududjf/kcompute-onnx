import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


class MulOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_in_tensor_1 { float in_tensor_1[]; };
layout (binding = 1) buffer buf_in_tensor_2 { float in_tensor_2[]; };
layout (binding = 2) buffer buf_out_tensor { float out_tensor[]; };
layout (constant_id = 0) const float size_x_1f = 0;
layout (constant_id = 1) const float size_y_1f = 0;
layout (constant_id = 2) const float size_z_1f = 0;
layout (constant_id = 3) const float size_x_2f = 0;
layout (constant_id = 4) const float size_y_2f = 0;
layout (constant_id = 5) const float size_z_2f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    uint size_x_1 = uint(size_x_1f);
    uint size_y_1 = uint(size_y_1f);
    uint size_z_1 = uint(size_z_1f);
    uint size_x_2 = uint(size_x_2f);
    uint size_y_2 = uint(size_y_2f);
    uint size_z_2 = uint(size_z_2f);
    uint stride_y_1 = size_z_1;
    uint stride_x_1 = size_y_1 * stride_y_1;
    uint stride_y_2 = size_z_2;
    uint stride_x_2 = size_y_2 * stride_y_2;
    uint stride_y = max(size_z_1, size_z_2);
    uint stride_x = max(size_y_1, size_y_2) * stride_y;
    uint x_1 = min(gx, size_x_1 - 1);
    uint y_1 = min(gy, size_y_1 - 1);
    uint z_1 = min(gz, size_z_1 - 1);
    uint x_2 = min(gx, size_x_2 - 1);
    uint y_2 = min(gy, size_y_2 - 1);
    uint z_2 = min(gz, size_z_2 - 1);
    uint p_1 = x_1 * stride_x_1 + y_1 * stride_y_1 + z_1;
    uint p_2 = x_2 * stride_x_2 + y_2 * stride_y_2 + z_2;
    out_tensor[gx * stride_x + gy * stride_y + gz] = in_tensor_1[p_1] * in_tensor_2[p_2];
}''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MulOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MulOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "MulOp requires 2 inputs"

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
        assert len(input_tensors) == 2, "MulOp requires 2 inputs"
        input_1 = input_tensors[0][0]
        input_2 = input_tensors[1][0]
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
                assert new_shape_1[i] == new_shape_2[i], f"MulOp requires input {i} of the same shape"
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
            size_x_1 = np.prod(new_shape_1[:-2])
            size_y_1 = new_shape_1[-2]
            size_z_1 = new_shape_1[-1]
            size_x_2 = np.prod(new_shape_2[:-2])
            size_y_2 = new_shape_2[-2]
            size_z_2 = new_shape_2[-1]

        size = np.prod(output_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        workgroup = (max(size_x_1, size_x_2), max(size_y_1, size_y_2), max(size_z_1, size_z_2))
        updated_algorithms.append(self.manager.algorithm([new_in_1, new_in_2, tensor_out],
                                                         self.compiled_shader,
                                                         workgroup,
                                                         [size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2],
                                                         []))

        return [(tensor_out, output_shape)]

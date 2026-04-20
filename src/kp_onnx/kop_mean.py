import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


class MeanOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.add_shader = compile_source('''
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
    out_tensor[gx * stride_x + gy * stride_y + gz] = (in_tensor_1[p_1] + in_tensor_2[p_2]);
}''')
        self.scale_shader = compile_source('''
#version 450
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;
layout (binding=0) buffer buf_in  { float in_tensor[]; };
layout (binding=1) buffer buf_out { float out_tensor[]; };
layout (constant_id=0) const float scale = 1.0;
void main() {
    uint gx = gl_GlobalInvocationID.x;
    out_tensor[gx] = in_tensor[gx] * scale;
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MeanOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MeanOp({device_name})"

    def run(self, *inputs):
        # 单输入：直接拷贝返回
        if len(inputs) == 1:
            return [inputs[0].astype(np.float32, copy=True)]

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
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
        if len(input_tensors) == 1:
            return [input_tensors[0]]

        def _binary_add_plan(x, y):
            input_1, shape_1 = x[0], list(x[1])
            input_2, shape_2 = y[0], list(y[1])
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
            for j in range(len(new_shape_1)):
                a, b = new_shape_1[j], new_shape_2[j]
                if a == 1:
                    output_shape.append(b)
                elif b == 1:
                    output_shape.append(a)
                else:
                    assert a == b, f"MeanOp requires input {j} of the same shape"
                    output_shape.append(a)

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

            size = int(np.prod(output_shape)) if len(output_shape) > 0 else 1
            output_tensor = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_tensors.append(output_tensor)

            workgroup = (max(size_x_1, size_x_2), max(size_y_1, size_y_2), max(size_z_1, size_z_2))
            updated_algorithms.append(self.manager.algorithm(
                [new_in_1, new_in_2, output_tensor],
                self.add_shader,
                workgroup,
                [size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2],
                []
            ))

            return output_tensor, output_shape

        cur = input_tensors[0]
        for i in range(1, len(input_tensors)):
            cur = _binary_add_plan(cur, input_tensors[i])

        sum_tensor, out_shape = cur

        scale = float(1.0 / len(input_tensors))
        output_size = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        tensor_out = self.manager.tensor(np.zeros(output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        algo_scale = self.manager.algorithm(
            [sum_tensor, tensor_out],
            self.scale_shader,
            (output_size, 1, 1),
            [scale],
            []
        )
        updated_algorithms.append(algo_scale)

        return [(tensor_out, out_shape)]

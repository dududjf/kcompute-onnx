import numpy as np
import kp
from .shader_utils import compile_source, broadcast_to


class WhereOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_condition { float condition[]; };
layout (binding = 1) buffer buf_x { float x[]; };
layout (binding = 2) buffer buf_y { float y[]; };
layout (binding = 3) buffer buf_out_tensor { float out_tensor[]; };
layout (constant_id = 0) const float size_x_cf = 0;
layout (constant_id = 1) const float size_y_cf = 0;
layout (constant_id = 2) const float size_z_cf = 0;
layout (constant_id = 3) const float size_x_xf = 0;
layout (constant_id = 4) const float size_y_xf = 0;
layout (constant_id = 5) const float size_z_xf = 0;
layout (constant_id = 6) const float size_x_yf = 0;
layout (constant_id = 7) const float size_y_yf = 0;
layout (constant_id = 8) const float size_z_yf = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    uint size_x_c = uint(size_x_cf);
    uint size_y_c = uint(size_y_cf);
    uint size_z_c = uint(size_z_cf);
    uint size_x_x = uint(size_x_xf);
    uint size_y_x = uint(size_y_xf);
    uint size_z_x = uint(size_z_xf);
    uint size_x_y = uint(size_x_yf);
    uint size_y_y = uint(size_y_yf);
    uint size_z_y = uint(size_z_yf);
    uint stride_y_c = size_z_c;
    uint stride_x_c = size_y_c * stride_y_c;
    uint stride_y_x = size_z_x;
    uint stride_x_x = size_y_x * stride_y_x;
    uint stride_y_y = size_z_y;
    uint stride_x_y = size_y_y * stride_y_y;
    uint stride_y = max(max(size_z_c, size_z_x), size_z_y);
    uint stride_x = max(max(size_y_c, size_y_x), size_y_y) * stride_y;
    uint x_c = min(gx, size_x_c - 1);
    uint y_c = min(gy, size_y_c - 1);
    uint z_c = min(gz, size_z_c - 1);
    uint x_x = min(gx, size_x_x - 1);
    uint y_x = min(gy, size_y_x - 1);
    uint z_x = min(gz, size_z_x - 1);
    uint x_y = min(gx, size_x_y - 1);
    uint y_y = min(gy, size_y_y - 1);
    uint z_y = min(gz, size_z_y - 1);
    uint p_c = x_c * stride_x_c + y_c * stride_y_c + z_c;
    uint p_x = x_x * stride_x_x + y_x * stride_y_x + z_x;
    uint p_y = x_y * stride_x_y + y_y * stride_y_y + z_y;
    out_tensor[gx * stride_x + gy * stride_y + gz] = (condition[p_c] != 0.0) ? x[p_x] : y[p_y];
}''')

    def __repr__(self):
        return f"WhereOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
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
        assert len(input_tensors) == 3, "WhereOp requires 3 inputs (condition, x, y)"
        condition = input_tensors[0][0]
        x = input_tensors[1][0]
        y = input_tensors[2][0]
        shape_c = input_tensors[0][1]
        shape_x = input_tensors[1][1]
        shape_y = input_tensors[2][1]

        max_dims = max(len(shape_c), len(shape_x), len(shape_y))

        new_shape_c = [1] * (max_dims - len(shape_c)) + shape_c
        new_shape_x = [1] * (max_dims - len(shape_x)) + shape_x
        new_shape_y = [1] * (max_dims - len(shape_y)) + shape_y

        output_shape = []
        for i in range(max_dims):
            max_size = max(new_shape_c[i], new_shape_x[i], new_shape_y[i])
            output_shape.append(max_size)

        new_condition = condition
        algorithms_c, next_tensors_c = [], []
        if output_shape[:-2] != new_shape_c[:-2] and not all(e == 1 for e in new_shape_c[:-2]):
            final_shape_c = output_shape[:-2] + list(new_shape_c[-2:])
            new_condition = broadcast_to(condition, new_shape_c, final_shape_c, algorithms_c, next_tensors_c, self.manager)
            updated_algorithms.extend(algorithms_c)
            new_shape_c = final_shape_c

        new_x = x
        algorithms_x, next_tensors_x = [], []
        if output_shape[:-2] != new_shape_x[:-2] and not all(e == 1 for e in new_shape_x[:-2]):
            final_shape_x = output_shape[:-2] + list(new_shape_x[-2:])
            new_x = broadcast_to(x, new_shape_x, final_shape_x, algorithms_x, next_tensors_x, self.manager)
            updated_algorithms.extend(algorithms_x)
            new_shape_x = final_shape_x

        new_y = y
        algorithms_y, next_tensors_y = [], []
        if output_shape[:-2] != new_shape_y[:-2] and not all(e == 1 for e in new_shape_y[:-2]):
            final_shape_y = output_shape[:-2] + list(new_shape_y[-2:])
            new_y = broadcast_to(y, new_shape_y, final_shape_y, algorithms_y, next_tensors_y, self.manager)
            updated_algorithms.extend(algorithms_y)
            new_shape_y = final_shape_y

        if len(new_shape_c) == 1:
            size_x_c = new_shape_c[0]
            size_y_c = 1
            size_z_c = 1
        elif len(new_shape_c) == 2:
            size_x_c = new_shape_c[0]
            size_y_c = new_shape_c[1]
            size_z_c = 1
        else:
            size_x_c = np.prod(new_shape_c[:-2])
            size_y_c = new_shape_c[-2]
            size_z_c = new_shape_c[-1]

        if len(new_shape_x) == 1:
            size_x_x = new_shape_x[0]
            size_y_x = 1
            size_z_x = 1
        elif len(new_shape_x) == 2:
            size_x_x = new_shape_x[0]
            size_y_x = new_shape_x[1]
            size_z_x = 1
        else:
            size_x_x = np.prod(new_shape_x[:-2])
            size_y_x = new_shape_x[-2]
            size_z_x = new_shape_x[-1]

        if len(new_shape_y) == 1:
            size_x_y = new_shape_y[0]
            size_y_y = 1
            size_z_y = 1
        elif len(new_shape_y) == 2:
            size_x_y = new_shape_y[0]
            size_y_y = new_shape_y[1]
            size_z_y = 1
        else:
            size_x_y = np.prod(new_shape_y[:-2])
            size_y_y = new_shape_y[-2]
            size_z_y = new_shape_y[-1]

        size = np.prod(output_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (max(max(size_x_c, size_x_x), size_x_y),
                     max(max(size_y_c, size_y_x), size_y_y),
                     max(max(size_z_c, size_z_x), size_z_y))

        updated_algorithms.append(self.manager.algorithm([new_condition, new_x, new_y, tensor_out],
                                                         self.compiled_shader,
                                                         workgroup,
                                                         [size_x_c, size_y_c, size_z_c,
                                                          size_x_x, size_y_x, size_z_x,
                                                          size_x_y, size_y_y, size_z_y],
                                                         []))

        return [(tensor_out, output_shape)]
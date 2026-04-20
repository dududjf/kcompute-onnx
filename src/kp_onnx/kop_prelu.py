import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


class PReLUOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_in_tensor_data  { float in_tensor_data[]; };
layout (binding = 1) buffer buf_in_tensor_slope { float in_tensor_slope[]; };
layout (binding = 2) buffer buf_out_tensor      { float out_tensor[]; };
layout (constant_id = 0) const float size_y_data = 0;
layout (constant_id = 1) const float size_z_data = 0;
layout (constant_id = 2) const float size_x_slope = 0;
layout (constant_id = 3) const float size_y_slope = 0;
layout (constant_id = 4) const float size_z_slope = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint stride_y_data = uint(size_z_data);
    uint stride_x_data = uint(size_y_data) * stride_y_data;
    uint idx_data = gx * stride_x_data + gy * stride_y_data + gz;

    uint x_slope = uint(size_x_slope);
    uint y_slope = uint(size_y_slope);
    uint z_slope = uint(size_z_slope);
    uint stride_y_slope = z_slope;
    uint stride_x_slope = y_slope * stride_y_slope;
    uint idx_slope = z_slope > 1 ? gz : 0;
    if (y_slope > 1) idx_slope += gy * stride_y_slope;
    if (x_slope > 1) idx_slope += gx * stride_x_slope;

    // Compute PReLU
    float data_val = in_tensor_data[idx_data];
    float slope_val = in_tensor_slope[idx_slope];
    out_tensor[idx_data] = data_val > 0 ? data_val : slope_val * data_val;
}''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PReLUOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        assert len(inputs) == 2, "PReLUOp requires 2 inputs: data and slope"

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
        assert len(input_tensors) == 2, "PReLUOp requires 2 inputs: data and slope"

        tensor_data = input_tensors[0][0]
        shape_data = input_tensors[0][1]
        tensor_slope = input_tensors[1][0]
        shape_slope = input_tensors[1][1]

        # Handle dimension alignment
        assert len(shape_slope) <= len(shape_data), "PReLUOp requires slope to have less or equal dimensions than data"
        if len(shape_slope) == 1 and shape_slope[0] > 1:
            new_shape_slope = []
            n = 0
            for d in shape_data:
                if d == shape_slope[0]:
                    new_shape_slope.append(d)
                    n += 1
                else:
                    new_shape_slope.append(1)
            assert n == 1, "PReLUOp requires slope to be one of data dimensions if it has only one dimension"
        elif len(shape_slope) < len(shape_data):
            new_shape_slope = [1] * (len(shape_data) - len(shape_slope)) + shape_slope
        else:
            new_shape_slope = shape_slope
        for dim_data, dim_slope in zip(shape_data, new_shape_slope):
            assert dim_slope == 1 or dim_slope == dim_data, \
                "PReLUOp requires each slope dimension to be one or equal to the corresponding data dimension"

        # Broadcast parameter slope if needed
        new_slope = tensor_slope
        if shape_data[:-2] != new_shape_slope[:-2] and not all(e == 1 for e in new_shape_slope[:-2]):
            final_shape_slope = shape_data[:-2] + list(new_shape_slope[-2:])
            new_slope = broadcast_to(tensor_slope, new_shape_slope, final_shape_slope,
                                     updated_algorithms, updated_tensors, self.manager)
            new_shape_slope = final_shape_slope

        # Determine size parameters
        if len(shape_data) == 1:
            size_x_data, size_y_data, size_z_data = shape_data[0], 1, 1
            size_x_slope, size_y_slope, size_z_slope = new_shape_slope[0], 1, 1
        elif len(shape_data) == 2:
            size_x_data, size_y_data, size_z_data = shape_data[0], shape_data[1], 1
            size_x_slope, size_y_slope, size_z_slope = new_shape_slope[0], new_shape_slope[1], 1
        else:
            size_x_data = np.prod(shape_data[:-2])
            size_y_data, size_z_data = shape_data[-2], shape_data[-1]
            size_x_slope = np.prod(new_shape_slope[:-2])
            size_y_slope, size_z_slope = new_shape_slope[-2], new_shape_slope[-1]

        # Create output tensor and algorithm
        size = np.prod(shape_data)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (size_x_data, size_y_data, size_y_data)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_data, new_slope, tensor_out],
            self.compiled_shader,
            workgroup,
            [size_y_data, size_z_data, size_x_slope, size_y_slope, size_z_slope],
            []
        ))

        return [(tensor_out, shape_data)]


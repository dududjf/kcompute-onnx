import kp
import numpy as np


class FlattenOp:
    def __init__(self, manager: kp.Manager, axis: int = 1):
        self.manager = manager
        self.axis = axis

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"FlattenOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"FlattenOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]
        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        shape_in = input_tensors[0][1]

        axis = self.axis
        if axis < 0:
            axis += len(shape_in)

        dim0 = np.prod(np.array(shape_in[:axis], dtype=np.int32))
        dim1 = np.prod(np.array(shape_in[axis:], dtype=np.int32))
        tensor_shape = [int(dim0), int(dim1)]

        tensor_out = input_tensors[0][0]
        return [(tensor_out, tensor_shape)]

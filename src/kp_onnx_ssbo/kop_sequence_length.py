import kp
import numpy as np


class SequenceLengthOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"SequenceLengthOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        sequence = inputs[0]
        input_tensors = [sequence]

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        tensor_out, output_shape = output_tensor_and_shape[0]
        output = tensor_out.data().reshape(output_shape)

        del tensor_out
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list, updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        sequence = input_tensors[0]
        length = len(sequence)

        length_array = np.zeros(1, dtype=np.float32)
        length_array[0] = length

        length_tensor = self.manager.tensor(length_array)
        updated_tensors.append(length_tensor)

        return [(length_tensor, [1])]


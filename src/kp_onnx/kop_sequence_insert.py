import kp
import numpy as np


class SequenceInsertOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"SequenceInsertOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []

        sequence_tensors = []
        for item in inputs[0]:
            numpy_in = item.reshape(-1).astype(np.float32) \
                if isinstance(item, np.ndarray) else np.array(item, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            sequence_tensors.append((tensor, list(item.shape) if isinstance(item, np.ndarray) else []))
        input_tensors.append(sequence_tensors)

        numpy_in = inputs[1].reshape(-1).astype(np.float32) \
            if isinstance(inputs[1], np.ndarray) else np.array(inputs[1], dtype=np.float32)
        tensor_to_insert = self.manager.tensor(numpy_in)
        insert_shape = list(inputs[1].shape) if isinstance(inputs[1], np.ndarray) else []
        input_tensors.append((tensor_to_insert, insert_shape))

        if len(inputs) > 2:
            input_tensors.append(inputs[2])

        updated_algorithms, updated_tensors = [], []
        output_tensors_and_shapes = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        outputs = []
        for tensor_out, output_shape in output_tensors_and_shapes:
            output = tensor_out.data().reshape(output_shape)
            outputs.append(output)

        for tensor, _ in sequence_tensors:
            del tensor
        del tensor_to_insert
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list, updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        sequence = input_tensors[0]
        tensor_to_insert, insert_shape = input_tensors[1]
        position = input_tensors[2] if len(input_tensors) > 2 else None

        result_sequence = []
        for tensor, shape in sequence:
            result_sequence.append((tensor, shape))

        if position is not None:
            if isinstance(position, np.ndarray):
                pos_data = position.astype(np.int32)
                pos = int(pos_data.flat[0]) if pos_data.size == 1 else int(pos_data[0])
            else:
                pos = int(position)
            insert_position = (pos + len(result_sequence)) % max(len(result_sequence), 1)
            result_sequence.insert(insert_position, (tensor_to_insert, insert_shape))
        else:
            result_sequence.append((tensor_to_insert, insert_shape))

        return result_sequence


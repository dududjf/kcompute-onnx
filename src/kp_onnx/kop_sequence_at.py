import numpy as np
import kp


class SequenceAtOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        return f"SequenceAtOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs[0]:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        numpy_in_pos = inputs[-1].reshape(-1).astype(np.int32) \
            if isinstance(inputs[-1], np.ndarray) else np.array(inputs[-1], dtype=np.int32)
        tensor_pos = self.manager.tensor(numpy_in_pos)
        input_tensors.append((tensor_pos, list(numpy_in_pos.shape) if isinstance(inputs[-1], np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
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
        pos = int(input_tensors[-1][0].data())
        if pos < 0:
            pos += len(input_tensors) - 1
        tensor_out = input_tensors[pos][0]
        output_shape = input_tensors[pos][1]
        updated_tensors.append(tensor_out)
        return [(tensor_out, output_shape)]

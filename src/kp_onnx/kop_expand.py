import numpy as np
import kp
from .shader_utils import broadcast_to


class ExpandOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ExpandOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ExpandOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "ExpandOp needs (input, target_shape)"
        data = inputs[0]
        shape = np.array(inputs[1], dtype=np.int32)

        data_tensor = self.manager.tensor(data.reshape(-1))
        shape_tensor = self.manager.tensor_t(shape, tensor_type=kp.TensorTypes.device)
        input_tensors = [(data_tensor, list(data.shape)), (shape_tensor, [shape.size])]

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
        assert len(input_tensors) == 2, "ExpandOp needs input tensor and target_shape"

        data_tensor = input_tensors[0][0]
        data_shape = input_tensors[0][1]
        target_shape_input = input_tensors[1][0].data().astype(np.int32).tolist()

        rank = max(len(data_shape), len(target_shape_input))
        padded_data_shape = [1] * (rank - len(data_shape)) + data_shape
        padded_target_shape = [1] * (rank - len(target_shape_input)) + target_shape_input

        new_shape = []
        for d, t in zip(padded_data_shape, padded_target_shape):
            assert (d == 1 or t == 1 or d == t), f"Cannot broadcast dimension {d} to {t}"
            new_shape.append(max(d, t))

        tensor_out = broadcast_to(data_tensor,
                                  padded_data_shape,
                                  new_shape,
                                  updated_algorithms,
                                  updated_tensors,
                                  self.manager)

        return [(tensor_out, new_shape)]

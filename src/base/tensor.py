import torch


def to_tensor_batch(batch, input_keys):
    for key in input_keys:
        if isinstance(batch[key], list) and isinstance(batch[key][0], torch.Tensor):
            batch[key] = torch.stack(batch[key], dim=1)
    return batch

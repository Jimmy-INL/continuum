import torch


class TorchStandardScaler:
    def fit_transform(self, x: torch.Tensor):
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x -= m
        x /= s
        return x
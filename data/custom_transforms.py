import torch
from torchvision.transforms import functional as ttf


__all__ = ["RandomRotate"]


class RandomRotate(torch.nn.Module):
    r"""Rotate the given image randomly with a given probability.
    All available values for the degree: [90, 180, 270]

    Args:
        p (float, optional): Probability of the image being rotated.
                             Default value is 0.5
    """

    def __init__(self, p: float = 0.5):
        super(RandomRotate, self).__init__()
        self._p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.rand(1)
        if r < self._p:
            return ttf.rotate(x, (int(r * 10) % 3) * 90)
        return x

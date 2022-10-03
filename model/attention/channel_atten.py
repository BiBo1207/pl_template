import torch.nn as nn
from util.model_utils import get_activation


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self._block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction,
                      kernel_size=1, padding=0, stride=1, bias=bias),
            get_activation(act_name='relu'),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel,
                      kernel_size=1, padding=0, stride=1, bias=bias),
            get_activation(act_name='sigmoid')
        )

    def forward(self, x):
        return x * self._block(x)


class CABlock(nn.Module):
    """
    ---conv--->act_func--->conv-------------->  + --->
                               ↓                ↑
                               ↓                ↑
                               ↓ ----> ca ----> ↑
    """
    def __init__(self, channel, ks, reduction, bias, act):
        super(CABlock, self).__init__()
        self._ca = CALayer(channel, reduction, bias)
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ks,
                      padding=ks // 2, stride=1, bias=bias),
            get_activation(act),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ks,
                      padding=ks // 2, stride=1, bias=bias),
        )

    def forward(self, x):
        res = self._block(x)
        res = self._ca(res)
        res += x
        return res


if __name__ == '__main__':
    from torchsummary import summary
    t = CABlock(64, 3, 16, False, 'relu').cuda()
    summary(t, (64, 256, 256))

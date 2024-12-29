import mindspore.nn as nn
import mindspore.ops as ops
from ms_snn.layer import LIFNode, VotingLayer
from ms_snn.sorrogate import ATan


class CIFAR10DVSNet(nn.Cell):
    def __init__(self, channels=128, class_num=10):
        super(CIFAR10DVSNet, self).__init__()
        self.conv_block1 = SpikingBlock(2, channels, 3, 1, 1)
        self.conv_block2 = SpikingBlock(channels, channels, 3, 1, 1)
        self.conv_block3 = SpikingBlock(channels, channels, 3, 1, 1)
        self.conv_block4 = SpikingBlock(channels, channels, 3, 1, 1)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Dense(channels * 8 * 8, 512)
        self.lif1 = LIFNode(surrogate_function=ATan())
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Dense(512, 10*class_num)
        self.lif2 = LIFNode(surrogate_function=ATan())
        self.voting = VotingLayer(10)
        self.net = nn.SequentialCell([self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4,
                                      self.flatten, self.dropout1, self.fc1, self.lif1, self.dropout2, self.fc2, self.lif2, self.voting])

    def construct(self, xs):
        xs = xs.transpose((1, 0, 2, 3, 4))
        ys = []
        for t in range(xs.shape[0]):
            x = xs[t]
            y = self.net(x)
            ys.append(y.unsqueeze(0))
        ys = ops.mean(ops.cat(ys, 0), 0)

        return ys


class SpikingBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SpikingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, 'pad', padding, has_bias=False)
        self.norm = nn.BatchNorm2d(
            out_channels, eps=1e-5, momentum=0.1, affine=True)
        self.lif = LIFNode(surrogate_function=ATan())
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def construct(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lif(x)
        x = self.pool(x)
        return x

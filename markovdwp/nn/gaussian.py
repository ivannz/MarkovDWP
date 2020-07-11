import torch
import torch.nn.functional as F

from .base import FreezableWeightBase

# subclass factorized Gaussian layers from cplxmodule
import cplxmodule.nn.relevance.real.base as gaussian


class LinearGaussian(gaussian.LinearGaussian, FreezableWeightBase):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.linear(input, self.weight_frozen, self.bias)


class BilinearGaussian(gaussian.BilinearGaussian, FreezableWeightBase):
    def forward(self, input1, input2):
        if not self.is_frozen():
            return super().forward(input1, input2)

        return F.bilinear(input1, input2, self.weight_frozen, self.bias)


class Conv1dGaussian(gaussian.Conv1dGaussian, FreezableWeightBase):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.conv1d(input, self.weight_frozen, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv2dGaussian(gaussian.Conv2dGaussian, FreezableWeightBase):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.conv2d(input, self.weight_frozen, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv3dGaussian(gaussian.Conv3dGaussian, FreezableWeightBase):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.conv3d(input, self.weight_frozen, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

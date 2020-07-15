import torch
import torch.nn.functional as F

from .base import FreezableWeightBase, BaseARD

# subclass factorized Gaussian layers from cplxmodule
import cplxmodule.nn.relevance.real.base as gaussian


class BasePenalizedGaussian(FreezableWeightBase, BaseARD):
    @property
    def penalty(self):
        r"""KL-divergence form the Standard Gaussian.

        Notes
        -----
        Computes the KL-divergence of $q_\theta(W)$ from $p(W) = N(W | 0, 1)$:
        $$
            \mathop{KL}\bigl(q_\theta(W) \| p(W)\bigr)
                = \mathbb{E}_{W \sim q_\theta}
                    \log \tfrac{q_\theta(W)}{p(W)}
                = \frac12 \sum_{ij} \bigl(
                    \sigma^2_{ij} + \mu_{ij}^2
                    - \log \sigma^2_{ij} - 1
                \bigr)
            \. $$
        """
        kl = self.weight * self.weight + torch.exp(self.log_sigma2)
        return 0.5 * (kl - self.log_sigma2 - 1)


class LinearGaussian(gaussian.LinearGaussian, BasePenalizedGaussian):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.linear(input, self.weight_frozen, self.bias)


class BilinearGaussian(gaussian.BilinearGaussian, BasePenalizedGaussian):
    def forward(self, input1, input2):
        if not self.is_frozen():
            return super().forward(input1, input2)

        return F.bilinear(input1, input2, self.weight_frozen, self.bias)


class Conv1dGaussian(gaussian.Conv1dGaussian, BasePenalizedGaussian):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.conv1d(input, self.weight_frozen, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv2dGaussian(gaussian.Conv2dGaussian, BasePenalizedGaussian):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.conv2d(input, self.weight_frozen, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv3dGaussian(gaussian.Conv3dGaussian, BasePenalizedGaussian):
    def forward(self, input):
        if not self.is_frozen():
            return super().forward(input)

        return F.conv3d(input, self.weight_frozen, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

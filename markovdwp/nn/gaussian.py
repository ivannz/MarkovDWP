import torch
import torch.nn.functional as F

from .base import PenalizedWeight, FreezableWeight


class BaseFreezableVariationalWeight(PenalizedWeight, FreezableWeight):
    def freeze(self):
        with torch.no_grad():
            stdev = F.softplus(self.log_var).sqrt()
            weight = torch.normal(self.weight, std=stdev)

        self.register_buffer('frozen_weight', weight)

    def reset_variational_parameters(self):
        self.log_var.data.normal_(-5, 0.1)  # from arxiv:1811.00596

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        log_sigma2 = torch.log(F.softplus(self.log_var))
        return log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)


class Linear(torch.nn.Linear, BaseFreezableVariationalWeight):
    """Linear layer with Gaussian Mean Field weight distribution."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_var = torch.nn.Parameter(
            torch.Tensor(*self.weight.shape))

        self.reset_variational_parameters()

    def forward(self, input):
        """Forward pass for the linear layer with the local reparameterization trick."""

        if self.is_frozen():
            return F.linear(input, self.frozen_weight, self.bias)

        s2 = F.linear(input * input, F.softplus(self.log_var), None)

        mu = super().forward(input)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class Conv2d(torch.nn.Conv2d, BaseFreezableVariationalWeight):
    """Convolutional layer with Gaussian Mean Field weight distribution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        self.log_var = torch.nn.Parameter(
            torch.Tensor(*self.weight.shape))

        self.reset_variational_parameters()

    def forward(self, input):
        """Forward pass for the Convolutional layer with the local reparameterization trick."""
        if self.is_frozen():
            return F.conv2d(input, self.frozen_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        s2 = F.conv2d(input * input, F.softplus(self.log_var), None,
                      self.stride, self.padding, self.dilation, self.groups)

        mu = super().forward(input)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))

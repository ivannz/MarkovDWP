import math
import torch


class VAMP(torch.nn.Module):
    """Varational Mixture of Posteriors prior by Tomczak and Welling (2017).

    URL
    ---
    https://arxiv.org/abs/1705.07120
    """
    def __init__(self, encoder, n_sample=50):
        super().__init__()
        self.encoder = encoder

        # pseudoinputs
        self.pseudoinputs = torch.nn.Parameter(
            torch.Tensor(n_sample, *encoder.input_shape)
        )

        self.reset_pseudoinputs()

    def reset_pseudoinputs(self):
        self.pseudoinputs.data.normal_(mean=0., std=0.01)

    @property
    def event_shape(self):
        return self.encoder.event_shape

    def rsample_from_index(self, index):
        """Draw diffenretiable variates from VAMP modes specified by `index`.
        """
        q = self.encoder(self.pseudoinputs[index.flatten()])
        return q.rsample().reshape(*index.shape, *self.event_shape)

    def rsample(self, sample_shape):
        r"""Draw diffenretiable variates from VAMP.

        Details
        -------
        Since VAMP is $\pi(z) = \frac1K \sum_k q(z\mid u_k)$, i.e. the mixture
        proibabilities are uniform and fixed, there is no need for REINFORCE.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        # index is uniformly random, and not learnable, thus no need to backprop
        #  through it and hence no need for reinforce gradients
        index = torch.randint(len(self.pseudoinputs),
                              size=(sample_shape.numel(),))

        return self.rsample_from_index(index.reshape(*sample_shape))

    @torch.no_grad()
    def sample(self, sample_shape):
        """Generate a `sample_shape` shaped sample from VAMP."""
        return self.rsample(sample_shape)

    def log_prob(self, value):
        r"""Differentiable log-probability of the VAMP prior.

        Details
        -------
        VAMP prior (Varational Mixture of Posteriors) is variant of emprirical
        Bayes prior learnt long with the VAE from the data. It the following
        mixture $
          \pi(z) = \tfrac1K \sum_k q(z \mid u_k)
        $ where $q(z|x)$ is the approximate posterior represented by the
        encoder network (with some distribution on output) and $u_k$ are
        learnable `pseudoinputs`, that detrmine the modes of the prior.

        The log-probability is log-sum-exp of log-probs of the VAE's encoder at
        each pseudoinput:
        $$
        \log \pi(z)
            = \log \Bigl(
                \sum_k \exp{ \{
                    \log q(z\mid u_k)
                \} }
            \Bigr) - \log K
            \,. $$
        """
        n_dim = len(self.event_shape)
        assert value.shape[-n_dim:] == self.event_shape

        # q has batch_shape `n_sample` (q_k(z) = q(z \mid u_k))
        q = self.encoder(self.pseudoinputs)

        # broadcast value so that `log_q` has shape `*batch_shape x n_sample`,
        # where `batch_shape` is the leading dimensions of `value`
        log_q = q.log_prob(value.unsqueeze(-n_dim-1))

        # vamp is \tfrac1K \sum_k q(z \mid u_k)
        # log-sum-exp-average along the pseudoinput dimension
        return log_q.logsumexp(dim=-1) - math.log(len(self.pseudoinputs))

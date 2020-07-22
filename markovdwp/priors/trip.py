import math
import torch

import torch.nn.functional as F

from torch.distributions import Independent, Normal


def softplus(*tensors):
    """Softplus all the cores."""
    return [*map(F.softplus, tensors)]


def packed_randn(*shape, dummy=0.):
    """Draw normal variates in packed format for the Gaussians in TRIP."""
    return torch.stack([
        F.pad(torch.randn(n), [0, max(shape) - n], value=dummy)
        for n in shape
    ], dim=1)


def gauss_log_prob(value, loc, log_scale):
    """Get the log-density of a univariate Gaussian.

    Details
    -------
    Why do people insist on using .pow(2)? `x * x` is generally faster than
    `x.pow(2)`, because the latter is a general purpose operation.

    Normal(loc, torch.exp(2 * log_scale)).log_prob(value)
    """
    z = (value - loc) * torch.exp(- log_scale)
    return -(z * z + math.log(2 * math.pi)) / 2 - log_scale


def trip_index_sample(k, cores):
    r"""Sample from the TRIP index distribution and get the log-prob.

    Details
    -------
    Consider a +ve tensor $A$ in TR-format. $\alpha$ -- mutliindex, and $
      G^{(k)} \in \mathbb{R}^{d_k \times r_k \times r_{k+1}}
    $ -- cores of the TR format.

    Use the cyclic property of the trace and the marginalization based on the
    mode-k tensor-vector product for $
      v\in \mathbb{R}^{d_k}
    $ and $
      \alpha_{-k} \in \prod_{i \neq k} [d_i]
    $:
    $$
    (A \times_k v)_{\alpha_{-k}}
        = \mathop{tr}\biggl\{
            \Bigl[ \sum_j G^{(k)}_j v_j \Bigr]
            \times \underbrace{
                \Bigl[ \prod_{i > k} G^{(i)}_{\alpha_i} \Bigr]
            }_{\text{tail}}
            \times \underbrace{
                \Bigl[ \prod_{i < k} G^{(i)}_{\alpha_i} \Bigr]
            }_{\text{head}}
        \biggr\}
        \,. $$
    """
    cores = softplus(*cores)

    # H_1 = I, H_k = \prod_{1 \leq s < k} \sum_j G^s_j
    heads = [None, cores[0].sum(dim=0)]
    for core in cores[1:]:
        heads.append(heads[-1] @ core.sum(dim=0))

    # the normalization constant is given by marginalizing all dimensions
    norm = heads.pop()

    sample, log_p = [], []
    for _ in range(k):
        # sampling using the "chain"-rule from d to 1 (in core order)
        # i_d \sim \tr( H_d G^d_{i_d}     )
        # i_k \sim \tr( H_k G^k_{i_k} T_k ), T_k = \prod_{s > k} G^s_{i_s}
        # i_1 \sim \tr(     G^1_{i_1} T_1 )
        index, tail = [], None
        for core, head in zip(cores[::-1], heads[::-1]):
            # `w_j` is the trace of {head @ core_j @ tail}
            if tail is None and head is None:
                w = core.diagonal(dim1=1, dim2=2).sum(dim=1)
            elif tail is None:
                w = torch.tensordot(head, core, [[0, 1], [2, 1]])
            elif head is None:
                w = torch.tensordot(core, tail, [[1, 2], [1, 0]])
            else:
                w = torch.tensordot(core, tail @ head, [[1, 2], [1, 0]])

            # sample from the core and update the tail
            ix = int(w.multinomial(1))
            tail = core[ix] if tail is None else core[ix] @ tail

            index.append(ix)

        # the trace of `tail` is the unnormalized prob of the sample
        log_p.append(tail.trace())
        sample.append(index[::-1])

    log_p = torch.stack(log_p, dim=0).log() - norm.trace().log()
    return torch.tensor(sample, device=log_p.device), log_p


def trip_sample(k, loc, log_scale, cores):
    # sample the indices
    index, log_p_index = trip_index_sample(k, cores)

    # create univariate Gaussians for each index
    location = loc[index, torch.arange(len(cores))]
    scale = torch.exp(log_scale)[index, torch.arange(len(cores))]

    return Independent(Normal(location, scale), 1), index, log_p_index


def trip_index_log_prob(index, cores):
    """log-prob of the TRIP index."""
    prob, norm = None, None
    for i, core in enumerate(softplus(*cores)):
        # \prod_{s \leq k} G^k_{\alpha_k}
        margin = core[index[:, i]]
        prob = margin if prob is None else prob @ margin

        # H_1 = I, H_k = \prod_{1 \leq s < k} \sum_j G^s_j
        norm = core.sum(dim=0) if norm is None else norm @ core.sum(dim=0)

    prob = prob.diagonal(dim1=1, dim2=2).sum(dim=1)
    log_prob = torch.clamp(prob, 1e-12).log()
    return log_prob - norm.trace().log()


def trip_log_prob(value, loc, log_scale, cores):
    r"""Compute the log-probability of a batch w.r.t TRIP.

    Details
    -------
    Let $
        p_k(z_k) = \bigl(
            \mathcal{N}(z_k\mid \mu_{kj}, \sigma^2_{kj})
        \bigr)_{j=1}^{d_k}
    $ be the vector of values of Gaussian univariate densities at $z_k$. Then
    $$
    p(z_k, \alpha_{-k})
        = \sum_{a_k} p(z_k, \alpha_k, \alpha_{-k})
        = \sum_{a_k} p(z_k \mid \alpha_k) p(\alpha_k, \alpha_{-k})
        = \frac{(A \times_k p_k(z_k))_{\alpha_{-k}}}{Z(A)}
        \,, $$
    where $Z(A)$ is the normalization constant $
      Z(A) = ((A \times_1 \mathbf{1}) \cdots \times_m \mathbf{1})
    $.
    """
    prob, norm = None, None
    for i, core in enumerate(softplus(*cores)):
        # batched univariate Gauss
        # dist = Normal(loc[:len(core), i], log_scale[:len(core), i].exp())
        # margin = torch.tensordot(dist.log_prob(z[:, [i]]).exp(), core, [[1], [0]])

        # M^k = \sum_j G^k_j p_j(z_k | \mu_{kj}, \sigma^2_{kj})
        log_prob = gauss_log_prob(
            value[:, [i]], loc[:len(core), i], log_scale[:len(core), i])
        margin = torch.tensordot(log_prob.exp(), core, [[1], [0]])

        # \prod_{s \leq k} M_s
        prob = margin if prob is None else prob @ margin

        # H_1 = I, H_k = \prod_{1 \leq s < k} \sum_j G^s_j
        norm = core.sum(dim=0) if norm is None else norm @ core.sum(dim=0)

    prob = prob.diagonal(dim1=1, dim2=2).sum(dim=1)
    log_prob = torch.clamp(prob, 1e-12).log()
    return log_prob - norm.trace().log()


class TRIP(torch.nn.Module):
    """Tensor Ring Induced Prior by Kuznetsov et al. (2019)

    References
    ----------
    .. [1] Kuznetsov, M., Polykovskiy, D., Vetrov, D. P., & Zhebrak, A.
       (2019). A prior of a Googol Gaussians: a tensor ring induced prior
       for generative models. In Advances in Neural Information Processing
       Systems (pp. 4102-4112).
    """
    def __init__(self, shape, ranks):
        if len(ranks) == len(shape):
            ranks = ranks[-1], *ranks
        assert ranks[0] == ranks[-1]

        super().__init__()
        self.shape, self.ranks = torch.Size(shape), torch.Size(ranks)

        # store locations and log-scales in 2d matrices in packed format
        self.location = torch.nn.Parameter(
            packed_randn(*self.shape, dummy=0.))
        self.log_scale = torch.nn.Parameter(
            packed_randn(*self.shape, dummy=float('-inf')))

        # cores are m_k x r_k x r_{k+1}
        self.cores = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(n, r0, r1))
            for n, r0, r1 in zip(shape, ranks, ranks[1:])
        ])

    def log_prob(self, value):
        return trip_log_prob(value, self.location,
                             self.log_scale, self.cores)

    @torch.no_grad()
    def sample(self, n_draws):
        d, ix, log_p = trip_sample(n_draws, self.location,
                                   self.log_scale, self.cores)
        return d.sample()

    def index_log_prob(self, index):
        return trip_index_log_prob(index, self.cores)

    @torch.no_grad()
    def index_sample(self, n_draws):
        ix, log_p = trip_index_sample(n_draws, self.cores)
        return ix

import math
import torch

import torch.nn.functional as F

from torch.distributions import Independent, Normal


def trip_index_sample(n_draws, cores, *, generator=None):
    r"""Sample from the TR categorical index distribution and get the log-prob.

    Details
    -------
    Consider a positive tensor $A$ in TR-format:
    $$
    A_\alpha
        = \mathop{tr}\Bigl\{ \prod_k G^{(k)}_{\alpha_k} \Bigr\}
        \,, $$
    where $
      \alpha \in \prod_k [d_k]
    $ is a multiindex, $
      [d_k] = 1,\,\cdots,\,d_k
    $ and $
      G^{(k)} \in \mathbb{R}^{d_k \times r_k \times r_{k+1}}
    $ are positive valued cores the TR format.

    Using the cyclic property of the trace and the marginalization based on
    the mode-k tensor-vector product for $
      v \in \mathbb{R}^{d_k}
    $ and $
      \alpha_{-k} \in \prod_{i \neq k} [d_i]
    $ is equal to:
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
    # H_1 = I, H_k = \prod_{1 \leq s < k} \sum_j G^s_j
    heads = [None, cores[0].sum(dim=0, keepdims=True)]
    for core in cores[1:]:
        heads.append(heads[-1] @ core.sum(dim=0, keepdims=True))

    # the normalization constant is given by marginalizing all dimensions
    norm = heads.pop().squeeze(0)

    # sampling using the "chain"-rule from m down to 1 (in reverse core order)
    # i_m \sim \tr( H_m G^m_{i_m}     )
    # i_k \sim \tr( H_k G^k_{i_k} T_k ), T_k = \prod_{s > k} G^s_{i_s}
    # i_1 \sim \tr(     G^1_{i_1} T_1 )
    index, tail = [], None
    for core, head in zip(cores[::-1], heads[::-1]):
        # `w_k` is the trace of {head @ core_k @ tail}
        if tail is None and head is None:
            # `w` is `d_m`
            w = core.diagonal(dim1=1, dim2=2).sum(dim=1)
            w = w.unsqueeze(0).expand(n_draws, -1)

        elif tail is None:
            # `H_m` is `1 x r_1 x r_m`, w_{i_m} = \tr( H_m G^m_{i_m} )
            w = torch.tensordot(head, core, [[1, 2], [2, 1]])
            w = w.expand(n_draws, -1)

        elif head is None:
            # `T_1` is `n_draws x r_2 x r_1`, w_{i_1} = \tr( G^1_{i_1} T_1 )
            w = torch.tensordot(tail, core, [[1, 2], [2, 1]])

        else:
            # `H_k` is `1 x r_1 x r_k`, `T_k` is `n_draws x r_{k+1} x r_1`
            # w_{q i_k} = \tr( H_k G^k_{i_k} T_{kq} ) = \tr( T_{kq} H_k G^k_{i_k} )
            w = torch.tensordot(tail @ head, core, [[2, 1], [1, 2]])

        # sample from the core and update the tail
        # `w` is `n_draws x d_k`, `ix` is `n_draws x 1`
        ix = w.multinomial(1, replacement=True, generator=generator)[:, 0]

        # T_{k-1 q} = G^k_{i_{kq}} T_{k q} is `n_draws x r_k x r_1`
        tail = core[ix] if tail is None else core[ix] @ tail
        index.append(ix)

    # the trace of `tail` is the unnormalized prob of the sample
    log_tail = tail.diagonal(dim1=1, dim2=2).sum(dim=1).log()
    return torch.stack(index[::-1], dim=1), log_tail - norm.trace().log()


def trip_index_log_prob(index, cores):
    """log-probability w.r.t. tensor ring categorical distribution."""
    prob, norm = None, None
    for i, core in enumerate(cores):
        # \prod_{s \leq k} G^k_{\alpha_k}
        margin = core[index[:, i]]
        prob = margin if prob is None else prob @ margin

        # H_1 = I, H_k = \prod_{1 \leq s < k} \sum_j G^s_j
        norm = core.sum(dim=0) if norm is None else norm @ core.sum(dim=0)

    log_prob = prob.diagonal(dim1=1, dim2=2).sum(dim=1).log()
    return log_prob - norm.trace().log()


class TRCategorical(torch.nn.Module):
    r"""Extremely high-dimensional categorical distribution parameterized by a
    tensor ring.

    Details
    -------
    Consider a positive tensor $A$ in Tensor Ring format:
    $$
    A_\alpha
        = \mathop{tr}\Bigl\{ \prod_k G^{(k)}_{\alpha_k} \Bigr\}
        \,, $$
    where $
      \alpha \in \prod_k [d_k]
    $ is a multiindex, $
      [d_k] = 1,\,\cdots,\,d_k
    $ and $
      G^{(k)} \in \mathbb{R}^{d_k \times r_k \times r_{k+1}}
    $ are positive valued cores the Tensor Ring format.

    Tensor Ring categorical posits that $
      p(\alpha) \propto A_\alpha
    $.
    """
    def __init__(self, shape, ranks):
        if len(ranks) == len(shape):
            ranks = ranks[-1], *ranks
        assert ranks[0] == ranks[-1]

        super().__init__()
        self.shape, self.ranks = torch.Size(shape), torch.Size(ranks)

        # cores are m_k x r_k x r_{k+1}
        self.log_cores = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(n, r0, r1))
            for n, r0, r1 in zip(shape, ranks, ranks[1:])
        ])

        self.reset_cores()

    def reset_cores(self):
        r"""Init the cores to an `almost` uniform distribution.

        Details
        -------
        Since $p_\alpha \propto A_\alpha$, by letting $
          G^{(k)}_j
            = \mathbf{1}_{r_k} \gamma_{k j} \mathbf{1}^\top_{r_{k+1}}
        $, we make sure that $
          A_\alpha = \prod_k r_k \gamma_{k \alpha_k}
        $ and $
          Z(A)
            = \sum_\alpha A_\alpha
            = \prod_k r_k \sum_{\alpha_k} \gamma_{k \alpha_k}
        $. Hence
        $$
        p(\alpha)
            = \prod_k \frac{
                \gamma_{k \alpha_k}
            }{
                \sum_{\alpha_k} \gamma_{k \alpha_k}
            }
            \,, $$
        with $\alpha \in \prod_k [d_k]$. So if $
          \gamma_{kj} = \frac1{d_k \sqrt{r_k}}
        $ then $
          p(\alpha) = \prod_k \frac1{d_k}
        $ and $
          A_\alpha = \prod_k \frac{\sqrt{r_k}}{d_k}
        $.
        """

        # ToDo: use tensor ring with 1/r_k mat-mat multipliers???
        for log_core in self.log_cores:
            dk, rk, _ = log_core.shape
            theta = math.log(math.expm1(1 / (dk * rk)))
            log_core.data.normal_(theta, std=0.1)

    @property
    def cores(self):
        """The on-the-fly computed +ve tensor ring cores via Softplus."""
        return [*map(F.softplus, self.log_cores)]

    def log_prob(self, index):
        """The log-probability w.r.t. the tensor-ring categorical.

        Parameters
        ----------
        index : torch.tensor
            The batch of indices to compute the log-probability of.

        Returns
        -------
        log_prob: torch.tensor
            The log-probability of each multi-index in the batch.
        """
        return trip_index_log_prob(index, self.cores)

    @torch.no_grad()
    def sample(self, n_draws):
        ix, log_p = self.sample_with_log_prob(n_draws)
        return ix

    def rsample(self, n_draws):
        raise NotImplementedError

    def sample_with_log_prob(self, n_draws):
        return trip_index_sample(n_draws, self.cores)


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
        super().__init__()

        self.index = TRCategorical(shape, ranks)

        # store locations and log-scales in 2d matrices in packed format
        self.location = torch.nn.Parameter(
            torch.Tensor(len(self.index.shape), max(self.index.shape)))
        self.logscale = torch.nn.Parameter(torch.Tensor(*self.location.shape))

        self.reset_parameters()

    def reset_parameters(self):
        """Draw normal variates in packed format for the Gaussians in TRIP."""
        self.location.data.zero_()
        self.logscale.data.fill_(-10.)  # float('-inf')
        for i, n in enumerate(self.index.shape):
            # location of each dimension is initialized to equispaced grid
            torch.linspace(-2, +2, n, out=self.location.data[i, :n])

            # scale is initialized to equispaced grid
            self.logscale.data[i, :n].fill_(-1.)

    @property
    def shape(self):
        # mirror shapes from the index
        return self.index.shape

    @property
    def cores(self):
        # expose the trip cores from the index
        return self.index.cores

    def log_prob(self, value):
        r"""Compute the log-probability of a batch of values w.r.t TRIP.

        Details
        -------
        Let $
          p_k(z_k) = \bigl(
              \mathcal{N}(z_k\mid \mu_{kj}, \sigma^2_{kj})
          \bigr)_{j=1}^{d_k}
        $ be the vector of values of Gaussian univariate densities at $z_k$.
        Then, since $
          p(z_k \mid \alpha) = p(z_k \mid \alpha_k) = p_k(z_k)_{\alpha_k}
        $, the density at $z$ is
        $$
        p(z)
            = \sum_\alpha p(\alpha) \prod_k p(z_k \mid \alpha)
            = \frac1{Z(A)} \sum_\alpha A_\alpha \prod_k p(z_k \mid \alpha)
            = \frac1{Z(A)} \mathop{tr}\Bigl\{
                \prod_k \sum_{\alpha_k} G^{(k)}_{\alpha_k} p_k(z_k)_{\alpha_k}
            \Bigr\}
            = \frac{(A \times_1 p_1(z_1) \cdots \times_m p_m(z_m))}{Z(A)}
            \,, $$
        where $Z(A)$ is the normalization constant $
          Z(A) = A \times_1 \mathbf{1} \cdots \times_m \mathbf{1}
        $. See details in `trip_index_sample()` for the mode-k tensor-vector
        product.

        Another example is:
        $$
        p(z_k, \alpha_{-k})
            = \sum_{\alpha_k} p(z_k, \alpha_k, \alpha_{-k})
            = \sum_{\alpha_k} p(z_k \mid \alpha_k) p(\alpha_k, \alpha_{-k})
            = \frac{(A \times_k p_k(z_k))_{\alpha_{-k}}}{Z(A)}
            \,. $$
        """
        assert value.dim() == 2

        prob, norm = None, None
        for i, core in enumerate(self.index.cores):
            # v_{kj} = \log p_j(z_k | \mu_{kj}, \sigma^2_{kj})
            loc = self.location[i, :len(core)]
            logstd = self.logscale[i, :len(core)]
            log_p = gauss_log_prob(value[:, [i]], loc, logstd)

            # M^k = \sum_j G^k_j e^{v_{kj} } -- sum-exp primitive would be nice
            margin = torch.tensordot(log_p.exp(), core, [[1], [0]])

            # \prod_{s \leq k} M_s
            prob = margin if prob is None else prob @ margin

            # H_1 = I, H_k = \prod_{1 \leq s < k} \sum_j G^s_j
            norm = core.sum(dim=0) if norm is None else norm @ core.sum(dim=0)

        prob = prob.diagonal(dim1=1, dim2=2).sum(dim=1)
        return torch.clamp(prob, 1e-12).log() - norm.trace().log()

    def rsample_from_index(self, index):
        """Sample from the chosen mode using the reparameterization.

        Details
        -------
        Should be used with conjunction with `.index.sample_with_log_prob` or
        `.index.sample`, e.g.:

        Python::
            trip = TRIP([11, 11, 11], [3, 3, 3])
            index, log_prob = trip.index.sample_with_log_prob(n_draws)
            values = trip.rsample_from_index(index)
        """
        assert index.dim() == 2

        # create univariate Gaussians for each row in the index
        rows_ = torch.arange(index.shape[-1], device=self.location.device)

        scale = self.logscale[rows_, index].exp()
        return self.location[rows_, index] + scale * torch.randn_like(scale)

    @torch.no_grad()
    def sample(self, n_draws):
        return self.rsample_from_index(self.index.sample(n_draws))

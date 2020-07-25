import math
import torch

import torch.nn.functional as F


def roll_left(*items, n):
    """Roll `items` `n` positions to the left so that `n`-th item is at 0"""
    _, pos = divmod(n, len(items))
    if pos == 0:
        return items
    return items[pos:] + items[:pos]


def trip_index_sample(n_draws, cores, *, generator=None):
    r"""Sample from the TR categorical index distribution and get the log-prob.

    Details
    -------
    Consider a positive tensor $A$ in TR-format:
    $$
    A_\alpha
        = \mathop{tr}\Bigl\{ \prod_k G^k_{\alpha_k} \Bigr\}
        \,, $$
    where $
      \alpha \in \prod_k [d_k]
    $ is a multiindex, $
      [d_k] = 1,\,\cdots,\,d_k
    $ and $
      G^k \in \mathbb{R}^{d_k \times r_k \times r_{k+1}}
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
            \Bigl[ \sum_j G^k_j v_j \Bigr]
            \times \underbrace{
                \Bigl[ \prod_{i > k} G^i_{\alpha_i} \Bigr]
            }_{\text{tail}}
            \times \underbrace{
                \Bigl[ \prod_{i < k} G^i_{\alpha_i} \Bigr]
            }_{\text{head}}
        \biggr\}
        \,. $$
    Let $
      \bar{G}^k =  \sum_{\alpha_k} G^k_{\alpha_k}
    $ denote the k-th `collapsed` core. Then the normalization constant is $
      Z(A)
        = \sum_\alpha A_\alpha
        = \mathop{tr}\Bigl\{ \prod_k \bar{G}^k \Bigr\}
    $, whence $
      p(\alpha) = \frac{A_\alpha}{Z(A)}
    $.
    """
    # Cycle the cores, so that the first one has the _smallest_ left rank dim.
    ranks = [core.shape[1] for core in cores]
    shift = min(range(len(ranks)), key=ranks.__getitem__)
    cores = roll_left(*cores, n=shift)

    # LTR-pass: H^1 = I, H^k = \prod_{s < k} \bar{G}^s = H^{k-1} \bar{G}^k
    heads, scale = [None], []
    for core in cores:
        bar_core = core.sum(dim=0, keepdims=True)
        head = bar_core if heads[-1] is None else heads[-1] @ bar_core

        # heads can be stabilized by their max-norm
        scale.append(float(head.max()))
        heads.append(head / scale[-1])

    # the normalization constant is given by marginalizing all dimensions
    # `norm` is H^{m+1} = \prod_k \bar{G}^k
    log_norm = heads.pop().squeeze(0).trace().log()
    log_norm_scale = sum(map(math.log, scale))

    # RTL-pass: T^m = I, T^k = \prod_{k < s} G^s_{i_s} = G^k_{i_k} T^{k+1}
    # sampling using the "chain"-rule from m down to 1 (in reverse core order)
    index, tail, dims, scale = [], None, [[1, 2], [2, 1]], []
    for core, head in zip(cores[::-1], heads[::-1]):
        # `H^k` is `1 x r_1 x r_k`, `T^k` is `n_draws x r_{k+1} x r_1`
        # `G^k` is `d_k x r_k x r_{k+1}`
        # w_{j i_k} = \tr( T^k_j H^k G^k_{i_k} ) -- prob of `i_k` on chain `j`
        with torch.no_grad():
            # weights `w` are used for sampling and are not backpropped through
            if tail is None and head is None:
                # single-core tensor ring -- just get the trace
                w = core.diagonal(dim1=1, dim2=2).sum(dim=1).unsqueeze(0)

            elif tail is None:
                # i_m \sim tr( H^m G^m_{i_m}     )
                w = torch.tensordot(head, core, dims=dims)

            elif head is None:
                # i_1 \sim tr(     G^1_{i_1} T^1 )
                w = torch.tensordot(tail, core, dims=dims)

            else:
                # i_k \sim tr( H^k G^k_{i_k} T^k )
                w = torch.tensordot(tail, head @ core, dims=dims)

        # make sure `w` is `n_draws x d_k` (has effect only if `tail` is None)
        w = w.expand(n_draws, -1)

        # prob of `i_k` is proportional to `w[.] = tr{head @ core[.] @ tail}`
        # differentiability and normalization of `w` not required for sampling
        ix = w.multinomial(1, replacement=True, generator=generator)[:, 0]

        # `ix` is `n_draws x 1`
        index.append(ix)

        # differentiably update the tail
        # `T^{k-1}_j = G^k_{i_{j k}} T^k_j` is `n_draws x r_k x r_1`
        tail = core[ix] if tail is None else core[ix] @ tail

        # track stabilizing scales, as `tail` dies off for extremely high-dim
        scale.append(tail.detach().flatten(1, -1).max(1).values)
        tail = tail / scale[-1].reshape(-1, 1, 1)

    # compute the log of the overall stabilizing mutliplier
    log_tail_scale = sum(map(torch.log, scale))

    # the trace of `tail` is the unnormalized prob of the sample
    log_tail = tail.diagonal(dim1=1, dim2=2).sum(dim=1).log()

    # reverse indices and roll back (`reversed` is clearer than [::-1])
    index = roll_left(*reversed(index), n=-shift)

    log_scale = log_tail_scale - log_norm_scale
    return torch.stack(index, dim=1), log_tail - log_norm + log_scale


def trip_index_log_marginals(cores):
    """marginal log-probability of each dimension in the tensor ring categorical
    distribution.
    """
    bars = [core.sum(dim=0) for core in cores]

    # LTR-pass: H^1 = I, H^k = \prod_{s < k} \bar{G}^s = H^{k-1} \bar{G}^k
    heads = [None]
    for bar in bars[:-1]:
        heads.append(bar if heads[-1] is None else heads[-1] @ bar)

    # RTL-pass: T^m = I, T^k = \prod_{k < s} \bar{G}^s = \bar{G}^k T^{k+1}
    tails = [None]
    for bar in bars[::-1]:  # reverse order!
        tails.append(bar if tails[-1] is None else bar @ tails[-1])

    # the normalization constant is given by marginalizing all dimensions
    norm = tails.pop().trace().log()  # T^0 = \prod_k \bar{G}^k

    log_p, dims = [], [[1, 2], [1, 0]]
    # k-th mariginal: w^k_j = \tr( H^k G^k_j T^k )
    for head, core, tail in zip(heads, cores, tails[::-1]):
        if head is None:
            # core is `d_1 x r_1 x r_2`, tail `r_2 x r_1`
            margin = torch.tensordot(core, tail, dims=dims)

        elif tail is None:
            # core is `d_m x r_m x r_1`, head `r_1 x r_m`
            margin = torch.tensordot(core, head, dims=dims)

        else:
            # core is `d_k x r_k x r_{k+1}`, head `r_1 x r_k`, tail `r_{k+1} x r_1`
            margin = torch.tensordot(core, tail @ head, dims=dims)

        log_p.append(margin.log() - norm)

    return log_p


def trip_index_log_prob(index, cores):
    """log-probability w.r.t. tensor ring categorical distribution."""
    prob, norm, scale = None, None, []
    for k, core in enumerate(cores):
        # H^1 = I, H^k = \prod_{s < k} \bar{G}^s = H^{k-1} \bar{G}^k
        norm = core.sum(dim=0) if norm is None else norm @ core.sum(dim=0)

        # P^k_j = \prod_{s \leq k} G^k_{i_{j k}}
        # `margin` is `n_samples x r_k x r_{k+1}`
        margin = core[index[:, k]]

        # `prob` is `n_samples x r_1 x r_{k+1}`
        prob = margin if prob is None else prob @ margin

        # stabilize
        scale.append(prob.detach().flatten(1, -1).max(1).values)
        prob = prob / scale[-1].reshape(-1, 1, 1)

    log_scale = sum(map(torch.log, scale))
    log_prob = prob.diagonal(dim1=1, dim2=2).sum(dim=1).log()
    return log_prob - norm.trace().log() + log_scale


def flatten_event_shape(tensor, event_shape, caller=''):
    """Check if rightmost dims of `tensor` match `event_shape` and then
    reshape into a matrix, by independently flattening leftmost and event
    dims.
    """
    tail = tensor.shape[-len(event_shape):]
    if tail != event_shape:
        origin = f' in {caller}' if caller else ''
        raise ValueError(f'Event shape mismatch`{origin}`. '
                         f'Expected `{event_shape}`. Got`{tail}`.')

    sample_shape = tensor.shape[:-len(event_shape)]
    return tensor.reshape(-1, torch.Size(event_shape).numel()), sample_shape


class TRCategorical(torch.nn.Module):
    r"""Extremely high-dimensional categorical distribution parameterized by a
    tensor ring.

    Parameters
    ----------
    shape : tuple
        Dimensions of the support of the tensor ring categorical distribution.

    ranks : tuple
        The ranks of the tensor cores used in the Tensor Ring parameterization.
        **Must** be the same length as `shape`. The higher the rank of each
        core the richer the dependency structure in the high-dimensional
        categorical distribution.

    event_shape : tuple, or None
        The shape of a single draw from this distribution, i.e. the event space
        within which the support sits. If `None`, then a single variate is a
        flat index vector of size `len(shape)`. Total number of elements taken
        by `event_shape` **must** be equal to `len(shape)`.

    Details
    -------
    Parameterizes a categorial distribution on $m$-dimensional support $
      \Omega = \prod_k [d_k]
    $ with $
      [d_k] = \{1,\,\cdots,\,d_k\}
    $ ($k=1..m$) by a positive-valued tensor $A$ (unnormalized): $
      p(\alpha) \propto A_\alpha
    $ where $
      \alpha \in \Omega
    $ is a single `event` (also multiindex). The tensor $A$ itself is stored
    only approximately via the Tensor Ring format:
    $$
    A_\alpha
        = \mathop{tr}\Bigl\{ \prod_k G^k_{\alpha_k} \Bigr\}
        \,, $$
    with positive-valued cores $
      G^k \in \mathbb{R}^{d_k \times r_k \times r_{k+1}}
    $. The higher the ranks $r_k$ the better the approximation of the tensor
    $A$, and thus less constrained the induced categorical distribution. For
    example, if $r_k = 1$ then all dimensions in $\Omega$ are independent: $
      p(\alpha) = \prod_k p(\alpha_k)
    $.
    """
    def __init__(self, shape, ranks, event_shape=None):
        assert len(ranks) == len(shape)

        # verify if `event_shape` is compatible with the tensor dimensionality.
        if event_shape is None:
            event_shape = (len(shape),)
        event_shape = torch.Size(event_shape)
        assert event_shape.numel() == len(shape)

        super().__init__()
        self.shape, self.ranks = torch.Size(shape), torch.Size(ranks)
        self.event_shape = event_shape

        # cores `G^k` are `d_k x r_k x r_{k+1}`, k=1..m
        ranks = *ranks, ranks[0]
        self.log_cores = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(d, r0, r1))
            for d, r0, r1 in zip(shape, ranks, ranks[1:])
        ])

        self.reset_cores()

    def reset_cores(self):
        r"""Init the cores to an `almost` uniform distribution.

        Details
        -------
        Since $p_\alpha \propto A_\alpha$, by letting $
          G^k_j
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
        """The log-probability density/mass w.r.t. the tensor ring categorical.

        Parameters
        ----------
        index : torch.tensor
            The sample of indices to compute the log-probability of. The
            rightmost dims must match the `event_shape` of this distribution.

        Returns
        -------
        log_prob: torch.tensor
            The log-probability of each multi-index in the sample. The shape of
            the returned tensor is equal to the leftmost dimes of the `value`
            (rightmost `event_shape` are consumed).
        """
        index, sample_shape = flatten_event_shape(index, self.event_shape)

        return trip_index_log_prob(index, self.cores).reshape(sample_shape)

    @torch.no_grad()
    def sample(self, sample_shape):
        """Generate a `sample_shape` shaped sample from the categorical."""
        index, log_p = self.sample_with_log_prob(sample_shape)
        return index

    def rsample(self, sample_shape):
        raise NotImplementedError

    def sample_with_log_prob(self, sample_shape):
        """Generate a `sample_shape` shaped sample from the categorical and
        return the log-probabilities of the variates in the sample.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        index, log_p = trip_index_sample(sample_shape.numel(), self.cores)
        index = index.reshape(*sample_shape, *self.event_shape)
        return index, log_p.reshape(sample_shape)

    def log_marginal(self):
        """The marginal log-probability mass of the tensor ring categorical."""
        return trip_index_log_marginals(self.cores)


def gauss_log_prob(value, loc, log_scale):
    """Get the log-density of a univariate Gaussian.

    Details
    -------
    Why do people insist on using .pow(2)? `x * x` is faster than `x.pow(2)`,
    because the latter is a general purpose operation.

    Normal(loc, torch.exp(2 * log_scale)).log_prob(value)
    """
    z = (value - loc) * torch.exp(- log_scale)
    return -(z * z + math.log(2 * math.pi)) / 2 - log_scale


class TRIP(torch.nn.Module):
    r"""Gaussian Tensor Ring Induced Prior by Kuznetsov et al. (2019).

    Parameters
    ----------
    shape : tuple
        Dimensions of the support of the tensor ring categorical distribution.

    ranks : tuple
        The ranks of the tensor cores used in the Tensor Ring parameterization.
        **Must** be the same length as `shape`. The higher the rank of each
        core the richer the dependency structure in the high-dimensional
        categorical distribution.

    event_shape : tuple, or None
        The shape of a single draw from this distribution, i.e. the event space
        within which the support sits. If `None`, then a single variate is a
        flat index vector of size `len(shape)`. Total number of elements taken
        by `event_shape` **must** be equal to `len(shape)`.

    Details
    --------
    This distribution is a ultra-high-dimensional Mixture of Gaussians with
    each component being a product of univariate Gaussians with location
    and scale determined by the component's index. Essentially
    $$
    p(z) = \nathbb{E}_{\alpha \sim p(\alpha)}
        p(z\mid \alpha)
        \,, $$
    with independent  $z_k$ conditional on $\alpha$: $
      p(z\mid \alpha)
        = \prod_k \mathcal{N}\bigl(
            z_k \mid \mu_{k \alpha_k}, \sigma^2_{k \alpha_k}
        \bigr)
    $ and the component-mixing distribution $p(\alpha)$. For the details on the
    underlying index distribution refer to `TRCategorical`.

    References
    ----------
    .. [1] Kuznetsov, M., Polykovskiy, D., Vetrov, D. P., & Zhebrak, A.
       (2019). A prior of a Googol Gaussians: a tensor ring induced prior
       for generative models. In Advances in Neural Information Processing
       Systems (pp. 4102-4112).
    """
    def __init__(self, shape, ranks, event_shape=None):
        super().__init__()

        self.index = TRCategorical(shape, ranks, event_shape)

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

    def log_prob(self, value):
        r"""Compute the log-probability density of Gaussian TRIP at `value`.

        Parameters
        ----------
        value : torch.tensor
            The sample of values to compute the log-probability of. The
            rightmost dims must match the `event_shape` of this distribution.

        Returns
        -------
        log_prob: torch.tensor
            The log-probability of each `event` in the sample. The shape of
            the returned tensor is equal to the leftmost dimes of the `value`
            (rightmost `event_shape` are consumed).

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
                \prod_k \sum_{\alpha_k} G^k_{\alpha_k} p_k(z_k)_{\alpha_k}
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
        # XXX semantics `batch_sahpe` maybe?
        value, sample_shape = flatten_event_shape(value, self.event_shape)

        prob, norm, scale, maxlogp = None, None, [], []
        for k, core in enumerate(self.index.cores):
            # H^1 = I, H^k = \prod_{s < k} \bar{G}^s = H^{k-1} \bar{G}^k
            norm = core.sum(dim=0) if norm is None else norm @ core.sum(dim=0)

            # v_{kj} = \log p_j(z_k | \mu_{kj}, \sigma^2_{kj})
            loc = self.location[k, :len(core)]
            logstd = self.logscale[k, :len(core)]
            log_p = gauss_log_prob(value[:, [k]], loc, logstd)

            # M^k = \sum_j G^k_j e^{v_{kj} } -- sum-exp primitive would be nice
            # max-norm as in log-sum-exp for stabilization
            maxlogp.append(log_p.detach().max(1, keepdims=True).values)
            # `log_p` is `n_samples x d_k`, `core` is `d_k x r_k x r_{k+1}`
            margin = torch.tensordot((log_p - maxlogp[-1]).exp(), core, dims=1)

            # P^k_j = \prod_{s \leq k} G^k_{i_{j k}}
            # `prob` is `n_samples x r_1 x r_{k+1}`
            prob = margin if prob is None else prob @ margin

            # stabilize running core-density product
            scale.append(prob.detach().flatten(1, -1).max(1).values)
            prob = prob / scale[-1].reshape(-1, 1, 1)

        log_scale = sum(map(torch.log, scale)) + sum(maxlogp).squeeze()
        log_prob = prob.diagonal(dim1=1, dim2=2).sum(dim=1).log()
        log_prob = log_prob - norm.trace().log() + log_scale

        return log_prob.reshape(sample_shape)

    def rsample_from_index(self, index):
        """Draw single variates from the Gaussian modes specified by `index`
        using location-scale reparameterization trick.

        Parameters
        ----------
        index : torch.tensor
            The sample of indices of the modes to draw single variates from.

        Details
        -------
        This method is seldom used alone. Most common use is in conjunction
        with `.index.sample_with_log_prob` or `.index.sample`, for mixed
        path-wise and score-function gradients.

        Example:

            >>> from markovdwp.priors.trip import TRIP
            >>> trip = TRIP([11, 11, 11], [3, 3, 3])
            >>> index, log_prob = trip.index.sample_with_log_prob([2, 2])
            >>> # `log_prob` can be used for REINFORCE
            >>> values = trip.rsample_from_index(index)
            >>> # `values` is differentiable
        """
        index, sample_shape = flatten_event_shape(index, self.event_shape)

        # create univariate Gaussians for each row in the index
        rows_ = torch.arange(index.shape[-1], device=self.location.device)

        scale = self.logscale[rows_, index].exp()
        samples = self.location[rows_, index] + scale * torch.randn_like(scale)
        return samples.reshape(*sample_shape, *self.event_shape)

    @torch.no_grad()
    def sample(self, sample_shape):
        """Generate a `sample_shape` shaped sample from the Gaussian TRIP."""
        return self.rsample_from_index(self.index.sample(sample_shape))

    # expose read-only properties from the underlying mode distribution
    @property
    def cores(self):
        return self.index.cores

    @property
    def shape(self):
        return self.index.shape

    @property
    def ranks(self):
        return self.index.ranks

    @property
    def event_shape(self):
        return self.index.event_shape

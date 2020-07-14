from ...utils.dicttools import flatten, aggregate, propagate


def prepare_log(details, level=5, delim='/'):
    prepared = aggregate(flatten(details, delim='.'), level=level, delim='.')
    return {k.replace('.', delim): v for k, v in prepared.items()}


def weighted_sum(terms, **coef):
    # 1. compute the final loss
    C = dict(propagate({'': 1.0, **coef}, terms, delim='.'))
    value = sum(v * C[k] for k, v in terms.items())

    # 2. return differentiable loss and its components as floats
    return value, {k: float(v) for k, v in terms.items()}


def collate(records):
    out = {}
    for record in records:
        for k, v in record.items():
            out.setdefault(k, []).append(v)
    return out


def linear(t, t0=0, t1=100, v0=1., v1=0.):
    tau = min(1., max(0., (t1 - t) / (t1 - t0)))
    return v0 * tau + v1 * (1 - tau)

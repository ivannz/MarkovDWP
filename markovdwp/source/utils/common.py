from collections import defaultdict


def propagate(lookup, G, prefix="", value=None):
    """Assign values propagating if necessary.

    Details
    -------
    Yields all prefixes of `G` with valued taken
    from `lookup` or propagated from parent.
    """
    # '' is a the parent of all nodes (except itself)
    if '' in G:
        yield from propagate(lookup, set(n for n in G if n), prefix, value)
        return

    # lookup (or inherit) the parent's value
    value = lookup.get(prefix, value)  # lookup.get(prefix, 1.) * value
    yield prefix, value

    # collect children of the current prefix (aka `parent`)
    children, prefix = {}, prefix + ('.' if prefix else '')
    for node in G:
        name, dot, child = node.partition('.')
        children.setdefault(prefix + name, set())
        if child:
            children[prefix + name].add(child)

    # propagate this parent's value to its children
    for prefix, G in children.items():
        yield from propagate(lookup, G, prefix, value)


def aggregate(dictionary, level=0, sep='.'):
    out, groups = {}, defaultdict(dict)
    for k, v in dictionary.items():
        key, _, child = k.partition(sep)
        if not child:
            out[key] = v
        else:
            groups[key][child] = v

    for name, group in groups.items():
        if level <= 0:
            out[name] = sum(group.values())
        else:
            group = aggregate(group, level - 1)
            out.update((name + sep + k, val) for k, val in group.items())
    return out


def flatten(dictionary, delim='.'):
    out = dict()
    for key in dictionary:
        value = dictionary[key]
        if isinstance(value, dict):
            nested = flatten(value, delim=delim)
            out.update((key + delim + k, val) for k, val in nested.items())

        else:
            out[key] = value

    return out


def prepare_log(details, level=5, delim='/'):
    prepared = aggregate(flatten(details, delim='.'), level=level, sep='.')
    return {k.replace('.', delim): v for k, v in prepared.items()}


def weighted_sum(terms, **coef):
    # 1. compute the final loss
    C = dict(propagate({'': 1.0, **coef}, terms))
    value = sum(v * C[k] for k, v in terms.items())

    # 2. return differentiable loss and its components as floats
    return value, {k: float(v) for k, v in terms.items()}


def collate(records):
    out = {}
    for record in records:
        for k, v in record.items():
            out.setdefault(k, []).append(v)
    return out

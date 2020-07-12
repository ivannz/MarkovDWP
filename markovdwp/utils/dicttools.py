import copy
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
    """Depth first redundantly flatten a nested dictionary.

    Arguments
    ---------
    dictionary : dict
        The dictionary to traverse and linearize.

    delim : str, default='.'
        The delimiter used to indicate nested keys.
    """
    out = dict()
    for key in dictionary:
        value = dictionary[key]
        if isinstance(value, dict):
            nested = flatten(value, delim=delim)
            out.update((key + delim + k, val) for k, val in nested.items())

        else:
            out[key] = value

    return out


def unflatten(dictionary, delim='.'):
    """Breadth first turn flattened dictionary into a nested one.

    Arguments
    ---------
    dictionary : dict
        The dictionary to traverse and linearize.

    delim : str, default='.'
        The delimiter used to indicate nested keys.
    """

    out = defaultdict(dict)
    # try to maintain curent order of the dictionary
    for key, value in dictionary.items():
        key, sep, sub_key = key.partition(delim)
        if sep:
            out[key][sub_key] = value
        else:
            out[key] = value

    for k, v in out.items():
        if isinstance(v, dict):
            out[k] = unflatten(v, delim)

    return dict(out)


def override(dictionary, **overrides):
    """Creates a shallow copy of the dictionary overrides. Supports
    scikit's `ParameterGrid` syntax for nested overrides.
    """
    if not overrides:
        return copy.deepcopy(dictionary)

    # split overrides into nested and local
    nested, local = defaultdict(dict), {}
    for key, value in overrides.items():
        key, delim, sub_key = key.partition('__')
        if delim:
            nested[key][sub_key] = value
        else:
            local[key] = value

    # override the non-nested items and possible introduce new ones
    out = {**dictionary, **local}
    for key, sub_params in nested.items():
        out[key] = override(dictionary.get(key, {}), **sub_params)

    return out

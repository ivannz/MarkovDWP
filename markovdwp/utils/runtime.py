import re
import importlib

from torch.utils.data import DataLoader


def get_class(name):
    """Parse the specified type-string, import it and return the type."""
    if isinstance(name, type):
        return name

    if not isinstance(name, str):
        raise TypeError(f"Expected a string, got {type(name)}.")

    # match and rsplit by "."
    match = re.fullmatch(r"^<(?:class)\s+'(?:(.*)\.)?([^\.]+)'>$", name)
    if match is None:
        raise ValueError(f"{name} is not a type identifier.")

    # import from built-ins if no module is specified
    module, name = (match.group(1) or "builtins"), match.group(2)
    return getattr(importlib.import_module(module), name)


def get_instance(*args, cls, **options):
    """Locate and create a `cls` instance."""
    return get_class(cls)(*args, **options)


def get_qualname(cls):
    """Get a qualifying name of a type."""
    x = get_class(cls)
    m, n = x.__module__, x.__name__
    return m + ('.' if m else '') + n


def get_datasets(datasets):
    return {
        name: get_instance(**klass)
        for name, klass in datasets.items()
    }


def get_dataloaders(datasets, feeds):
    return {
        feed: DataLoader(datasets[feed], **settings)
        for feed, settings in feeds.items()
    }

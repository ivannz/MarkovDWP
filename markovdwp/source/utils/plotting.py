import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def checkerboard(images, n_row, n_col, r=0.1):
    n_images, height, width, *color = images.shape

    lo, hi = images.min(), images.max()
    canvas = np.empty((n_row * height, n_col * width, *color))
    canvas[0::2, 0::2], canvas[0::2, 1::2] = lo + r * abs(lo), hi - r * abs(hi)
    canvas[1::2, 0::2], canvas[1::2, 1::2] = hi - r * abs(hi), lo + r * abs(lo)

    canvas = canvas.reshape(n_row, height, n_col, width, *color)
    return canvas.swapaxes(1, 2).reshape(-1, height, width, *color).copy()


def arrange(images, n_row, n_col):
    n_images, height, width, *color = images.shape

    canvas = checkerboard(images, n_row, n_col)
    canvas[:n_images] = images

    canvas = canvas.reshape(n_row, n_col, height, width, *color)
    canvas = canvas.swapaxes(1, 2)
    return canvas.reshape(n_row * height, n_col * width, *color)


def geometry(images, n_row=None, n_col=None, f_aspect=1.):
    n_images, height, width, *color = images.shape
    if n_row is None and n_col is None:
        ratio = width / (height * f_aspect)
        n_row = int(np.ceil(np.sqrt(n_images * ratio)))

    if n_row is None:
        n_row = (n_images + n_col - 1) // n_col

    if n_col is None:
        n_col = (n_images + n_row - 1) // n_row

    return n_row, n_col


def plot(images, ax=None, f_aspect=1., **kwargs):
    n_row, n_col = geometry(images, None, None, f_aspect)
    canvas = arrange(images, n_row, n_col)
    if canvas.ndim > 2 and canvas.shape[-1] < 2:
        canvas = canvas[..., 0]  # squeeze color dim only!

    ax = plt.gca() if ax is None else ax
    ax.imshow(canvas, **kwargs)

    # light grid between patches
    ax.add_collection(LineCollection(
        [((0, y / n_row), (1, y / n_row)) for y in range(1, n_row)] +
        [((x / n_col, 0), (x / n_col, 1)) for x in range(1, n_col)],
        zorder=20, alpha=0.5, linewidths=.5, colors='w',
        transform=ax.transAxes
    ))

    # annotate
    for y in range(n_row):
        ax.text(-0.02, (2 * y + 0.75) / (2 * n_row), f'{n_row - y:2d}',
                rotation='horizontal', horizontalalignment='left',
                transform=ax.transAxes, fontsize='x-small')
        ax.text(1.02, (2 * y + 0.75) / (2 * n_row), f'{n_row - y:2d}',
                rotation='horizontal', horizontalalignment='right',
                transform=ax.transAxes, fontsize='x-small')

    for x in range(n_col):
        ax.text((2 * x + 0.75) / (2 * n_col), 1.01, f'{x+1:2d}',
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes, fontsize='x-small')
        ax.text((2 * x + 0.75) / (2 * n_col), -.01, f'{x+1:2d}',
                rotation='horizontal', horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes, fontsize='x-small')

    return ax


def plot_slices(slices, cmap=plt.cm.coolwarm):
    assert isinstance(slices, torch.Tensor) and slices.dim() == 3

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120,
                           subplot_kw=dict(xticks=[], yticks=[]))

    images = slices.cpu().numpy()[..., np.newaxis]
    plot(images, ax=ax, f_aspect=16 / 9, cmap=cmap)
    fig.tight_layout()

    plt.close()

    return fig

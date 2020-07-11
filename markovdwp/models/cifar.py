from collections import OrderedDict

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d
from torch.nn import LeakyReLU, MaxPool2d, Flatten, Sequential

from cplxmodule.nn.relevance import Conv2dVD


class CIFARNetFeatures:
    """Feature block of CifarNet from the DWP paper."""
    def __new__(cls, *recipe, conv2d, bias=False):
        head, *rest = recipe
        features = [
            ('conv0', conv2d(head, rest[0], 7, stride=1, bias=bias)),
            ('norm0', BatchNorm2d(rest[0])),
            ('relu0', LeakyReLU()),
            ('maxpool', MaxPool2d(2))
        ]

        for j, (n, m) in enumerate(zip(rest, rest[1:]), 1):
            features.extend([
                (f'conv{j}', conv2d(n, m, 5, stride=1, bias=bias)),
                (f'norm{j}', BatchNorm2d(m)),
                (f'relu{j}', LeakyReLU()),
            ])

        # flatten instead of pooling
        features.append(('flatten', Flatten(1, -1)))

        return Sequential(OrderedDict(features))


class SourceCIFARNet:
    """Vanilla CifarNet from the DWP paper.

    Details
    -------
    Usually bias is disabled in layers just before batchnorm. We replicate
    the model here according to the specs listing in the paper.
    """
    Linear, Conv2d = Linear, Conv2d

    def __new__(cls, n_classes=10, k=1.):
        # input 3 and output 512 fixed
        *recipe, = map(int, [3, 128 * k, 256 * k, 256 * k, 512])
        features = CIFARNetFeatures(*recipe, conv2d=cls.Conv2d, bias=False)

        classifier = [
            ('fc1', cls.Linear(512, 512, bias=True)),
            ('bn1', BatchNorm1d(512)),
            ('relu', LeakyReLU()),
            ('linear', cls.Linear(512, n_classes, bias=True)),
        ]

        return Sequential(OrderedDict([
            ('features', features),
            ('classifier', Sequential(OrderedDict(classifier)))
        ]))


class CIFARNet:
    """CifarNet from the DWP paper.

    Details
    -------
    The model here deviates from the specs in the paper, since it
    is customary to disable bias in layers just before batchnorm.
    """
    Linear, Conv2d = Linear, Conv2d

    def __new__(cls, n_classes=10, k=1.):
        *recipe, = map(int, [3, 128 * k, 256 * k, 256 * k])
        features = CIFARNetFeatures(*recipe, conv2d=cls.Conv2d, bias=False)

        classifier = [
            ('fc1', cls.Linear(6400, 512, bias=False)),
            ('bn1', BatchNorm1d(512)),
            ('relu', LeakyReLU()),
            ('linear', cls.Linear(512, n_classes, bias=True)),
        ]

        return Sequential(OrderedDict([
            ('features', features),
            ('classifier', Sequential(OrderedDict(classifier)))
        ]))


class BayesCIFARNet(CIFARNet):
    Linear, Conv2d = Linear, Conv2dVD

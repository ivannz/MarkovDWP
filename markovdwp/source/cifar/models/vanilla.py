from collections import OrderedDict

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d
from torch.nn import LeakyReLU, MaxPool2d, Flatten, Sequential


class CIFARNet:
    """Vanilla CifarNet from the DWP paper.

    Details
    -------
    Usually bias is disabled in layers just before batchnorm. We replicate
    the model here according to the specs listing in the paper.
    """
    def __new__(cls, n_classes=10, k=1.):
        # input 3 and output 512 fixed
        head, *rest = map(int, [3, 128 * k, 256 * k, 256 * k, 512])
        features = [
            ('conv0', Conv2d(head, rest[0], 7, stride=1)),
            ('norm0', BatchNorm2d(rest[0])),
            ('relu0', LeakyReLU()),
            ('maxpool', MaxPool2d(2))
        ]

        for j, (n, m) in enumerate(zip(rest, rest[1:]), 1):
            features.extend([
                (f'conv{j}', Conv2d(n, m, 5, stride=1)),
                (f'norm{j}', BatchNorm2d(m)),
                (f'relu{j}', LeakyReLU()),
            ])

        # flatten instead of pooling
        features.append(('flatten', Flatten(1, -1)))

        classifier = [
            ('fc1', Linear(512, 512, bias=True)),
            ('bn1', BatchNorm1d(512)),
            ('relu', LeakyReLU()),
            ('linear', Linear(512, n_classes, bias=True)),
        ]

        return Sequential(OrderedDict([
            ("features", Sequential(OrderedDict(features))),
            ("classifier", Sequential(OrderedDict(classifier)))
        ]))

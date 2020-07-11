import os
import torch
import numpy as np

from PIL import Image
from PIL import UnidentifiedImageError


from torchvision.datasets.mnist import MNIST
from torchvision.datasets.utils import (download_and_extract_archive,
                                        verify_str_arg)


class notMNIST(MNIST):
    """`notMnist <http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html>`_ Dataset.

    Args:
        root (string): Root directory of the dataset whose ``processed''
            subdir contains torch binary files with the datasets.
        part (string, optional): Specifies which part of the dataset to use.
            Can be 'small' (hand-cleaned, 19k instances) or 'large' (uncleaned
            dataset, 500k instances).
        download (bool, optional): If true, downloads the dataset from
            the internet and puts it in root directory. If dataset is
            already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform
            that takes in the target and transforms it.
        train (bool,optional,compatibility): Ignored for this dataset.
    """
    resources = {
        'small': (
            'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz',
            'c9890a473a9769fda4bdf314aaf500dd',
            'notMNIST_small'
        ),
        'large': (
            'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz',
            '70a95b805ecfb6592c48e196df7c1499',
            'notMNIST_large'
        ),
    }
    classes = ['0 - A', '1 - B', '2 - C', '3 - D', '4 - E',
               '5 - F', '6 - G', '7 - H', '8 - I', '9 - J']

    def __init__(self, root, train=True, part='small', **kwargs):
        self.part = verify_str_arg(part, "part", tuple(self.resources.keys()))

        self.data_file = part + '.pt'
        self.training_file = self.data_file
        self.test_file = self.data_file
        super().__init__(root, train, **kwargs)

    def download(self):
        """Download and preprocess the notMNIST data if necessary exist."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        url, md5, dirname = self.resources[self.part]
        filename = url.rpartition('/')[2]
        if not os.path.isfile(os.path.join(self.raw_folder, filename)):
            download_and_extract_archive(
                url, filename=filename, md5=md5,
                download_root=self.raw_folder)

        # the images in notMNIST are letters A-J typeset with different fonts
        to_label = dict(zip("ABCDEFGHIJ", range(10)))
        root = os.path.join(self.raw_folder, dirname)

        # process and save as torch files
        tensors, labels = [], []
        root, classes, _ = next(os.walk(root))
        for klass in classes:
            images = []
            path, _, filenames = next(os.walk(os.path.join(root, klass)))
            for filename in filenames:
                try:
                    img = Image.open(os.path.join(path, filename))

                except UnidentifiedImageError as e:
                    print(str(e))
                    continue

                images.append(torch.from_numpy(np.array(img)))

            tensors.append(torch.stack(images, dim=0))
            labels.append(torch.full((len(images),), to_label[klass],
                                     dtype=torch.long))

        data, targets = torch.cat(tensors, dim=0), torch.cat(labels, dim=0)
        filename = os.path.join(self.processed_folder, self.data_file)
        with open(filename, 'wb') as f:
            torch.save((data, targets), f)

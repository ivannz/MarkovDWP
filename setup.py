from distutils.core import setup

setup(
    name='MarkovDWP',
    version='0.4',
    description='''Backend for experiments with Deep Weight Prior''',
    license='MIT License',
    packages=[
        'markovdwp',
        'markovdwp.nn',
        'markovdwp.priors',
        'markovdwp.priors.dwp',
        'markovdwp.source',
        'markovdwp.source.cifar',
        'markovdwp.source.cifar.models',
        'markovdwp.source.mnist',
        'markovdwp.source.mnist.models',
        'markovdwp.source.notmnist',
        'markovdwp.source.svhn',
        'markovdwp.source.dataset',
        'markovdwp.source.utils',
        'markovdwp.models',
        'markovdwp.utils',
        'markovdwp.utils.vendor',
        'markovdwp.utils.vendor.pytorch_lightning',
    ],
    install_requires=[
        'torch>=1.5',
        'torchvision',
        'cplxmodule',
        'pytorch-lightning',
        'sklearn',
    ]
)

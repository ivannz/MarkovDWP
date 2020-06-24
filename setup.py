from distutils.core import setup

setup(
    name='MarkovDWP',
    version='0.2',
    description='''Backend for experiments with Deep Weight Prior''',
    license='MIT License',
    packages=[
        'markovdwp',
        'markovdwp.priors',
        'markovdwp.source',
        'markovdwp.source.cifar',
        'markovdwp.source.cifar.models',
        'markovdwp.source.mnist',
        'markovdwp.source.mnist.models',
        'markovdwp.source.utils',
        'markovdwp.utils',
    ],
    install_requires=[
        'torch>=1.4',
        'torchvision',
        'pytorch-lightning',
        'sklearn',
    ]
)

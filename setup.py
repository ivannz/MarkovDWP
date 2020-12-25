from setuptools import setup

setup(
    name='MarkovDWP',
    version='2020.12',
    description='''Backend for experiments with Deep Weight Prior''',
    license='MIT License',
    packages=[
        'markovdwp',
        'markovdwp.nn',
        'markovdwp.priors',
        'markovdwp.source',
        'markovdwp.source.kernel',
        'markovdwp.source.datasets',
        'markovdwp.models',
        'markovdwp.models.dwp',
        'markovdwp.runtime',
        'markovdwp.runtime.utils',
        'markovdwp.utils',
        'markovdwp.utils.vendor',
        'markovdwp.utils.vendor.pytorch_lightning',
    ],
    install_requires=[
        'torch>=1.5',
        'torchvision',
        'cplxmodule',
        'pytorch-lightning>=1.1',
        'sklearn',
        'wandb',
    ]
)

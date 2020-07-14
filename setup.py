from setuptools import setup

setup(
    name='MarkovDWP',
    version='0.5',
    description='''Backend for experiments with Deep Weight Prior''',
    license='MIT License',
    packages=[
        'markovdwp',
        'markovdwp.nn',
        'markovdwp.priors',
        'markovdwp.priors.utils',
        'markovdwp.source',
        'markovdwp.source.kernel',
        'markovdwp.source.datasets',
        'markovdwp.source.utils',
        'markovdwp.models',
        'markovdwp.models.dwp',
        'markovdwp.runtime',
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

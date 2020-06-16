from distutils.core import setup

setup(
    name='MarkovDWP',
    version='0.1',
    description='''Backend for experiments with Deep Weight Prior''',
    license='MIT License',
    packages=[
        'markovdwp',
    ],
    install_requires=[
        'torch>=1.4',
        'pytorch-lighting'
    ]
)

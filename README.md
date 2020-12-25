# Markov Deep Weight Prior

The reviews of the Deep Weight Prior, proposed by [Atanov et al. (2019)](https://openreview.net/forum?id=ByGuynAct7), were mostly concerned with modelling issues: **factorization** and **layer and channel independence**, since there is ample evidence regarding the dependence of convolutional kernel slices within layers and between them, e.g. [Olah et al.(2020)](https://distill.pub/2020/circuits/zoom-in).

In this project we set out to find a solution, a hierarchical model, that would enable generation of interdependent slices for the filter bank of a single convolutional layer. Having achieved a substantial result, we could explore further tentative directions: generative model for entire layers, progressive growth of neural networks in architecture search.

## Setup

### Environment
The base stack is `python>=3.7`, `torch>=1.5` as the backend, `wandb` for reporting and `pytorch-lighting=0.9.1` as the framework.

For example, you can setup the following *conda* environment
```bash
conda create -n pysandbox "python>=3.7" pip cython numpy mkl numba scipy scikit-learn \
jupyter ipython pytest "conda-forge::compilers>=1.0.4" conda-forge::llvm-openmp \
matplotlib pytorch::pytorch pytorch::torchvision pytorch::torchaudio "cudatoolkit=10.2"

conda activate pysandbox
```

### Reporting Backend

Before using the package it is necessary to insall and setup Weights and Biases. Please follow these [quickstart instructions](https://docs.wandb.com/quickstart). The essential steps are provided below

```bash
pip install wandb

wandb login
```

### Package installation

To install unzip or clone the repository of the package
```bash
unzip MarkovDWP-master.zip
# git clone https://github.com/ivannz/MarkovDWP.git MarkovDWP

# local module installation
cd ./MarkovDWP-master

pip install -e .

chmod +x ./scripts/_source.sh ./scripts/source.sh
chmod +x ./scripts/dataset.sh ./scripts/sweep.sh

# run tests to ensure proper dependecies
pytest
```

Run the following to see if the package has been installed properly
```bash
cd ./MarkovDWP-master

# download cifar10/100 datasets and validate the pipeline
CUDA_VISIBLE_DEVICES=0 python -m markovdwp.source ./experiments/configs/prepare-cifar10.json
CUDA_VISIBLE_DEVICES=0 python -m markovdwp.source ./experiments/configs/prepare-cifar100.json
```

## Brief overview of the MarkovDWP package

The package implements prior distributions, convolutional networks, ELBO and IWAE objectives, and training and logging methods automated experimentation pipeline. The objective of the package is replicate the original DWP paper and to provide reusable implementations of key objects: `ImplcitPrior`, `TensoRingPrior`, `SingleKernelDataset` and etc.

The repo has the following structure:
* `data` -- the folder where the data is expected to reside by default
* `experiments` -- the directory where experiments are designed and their prototypes kept
* `markovdwp` -- the package with method implementations and the core experimentation backend
* `scripts` -- the bash scripts that facilitate automated experiments


## Large scale experiments

The package provides interface for large scale automated experiments. To achieve this it was necessary to automate model, dataset and other object creation and allow to re-specify them without human intervention between experiments.

### The `class-spec` dictionary

The core object definition unit in the configuration JSON is the `class spec` dictionary. It is responsible for instantiation and enables on-the-fly replacement of objects. For example, it enables easier grid search over implemented models, datasets, or other objects. The class spec is a dictionary with a required key `cls`, which specified the [qualified name](https://www.python.org/dev/peps/pep-3155/#id6) of the desired object represented in the form of a string with format:
> `"<class 'python.path.to.MyObjectClass'>`

For example, the following class-spec represents a linear layers with 16 input and 32
output neurons without bias term:
```python
class_spec = {
    'cls': "<class 'torch.nn.modules.linear.Linear'>",
    'in_features': 16,
    'out_features': 32,
    'bias': False
}
```
Internally class-specs are resolved into object instances via the following
```python
from markovdwp.utils.runtime import get_instance

# get instance from the class-spec
layer = get_instance(**class_spec)
print(layer)
# >>> Linear(in_features=16, out_features=32, bias=False)
```
The higher-level logic is
1. use `importlib` to find and import the object specified in the mandatory field `cls`
2. construct the located object with the keyword arguments taked from all fields of the class-spec other than `cls`.

To build class-specs of new objects one can do:
```python
import torch

# A qualified name lets you re-import the exact same object
print(str(torch.nn.ConvTranspose2d))
# >>> <class 'torch.nn.modules.conv.ConvTranspose2d'>

class_spec = {
    'cls': str(torch.nn.ConvTranspose2d),
    'in_channels': 16,
    'out_channels': 32,
    'kernel_size': 3,
    'stride': 1,
    'bias': True
}
```


#### partial class-spec

Partial class-spec is just the string which would have been under the `cls` field of the full class-spec. This format enables to split class from parameters and is used in `experiments/vae.py`, but rarely elsewhere.

### Specification of experiments

Every experiment is defined through a `manifest`, which is a `JSON` file with a special structure, that specifies the model, its parameters, the dataset, the dataloaders, the training procedure and the applied method.

The JSON has the following mandatory fields:
* `dataset` -- specifies the datasets used in the training procedure
* `feeds` -- specified the parameters of the dataloaders used to randomly draw batches from the specified dataset
* `trainer` -- parameters of the training loop, e.g. `max number of epochs`, `gradient clipping`, `validation frequency`, etc.
* `options` -- parameters of the gradient descent training, e.g. the `learning rate`, and coefficients in the objective function, which is to be minimized

Each particular kind of experiment may require additional fields. For example, the `model` field, which keeps the specification of the model that is to be trained, is not used by `experiments/vae.py`
* `model` is a class-spec dictionary of the model used for training

#### dataset

`datasets` is dictionary of class-specs, each key of which specifies a dataset. The key becomes the name assigned to the dataset.

The following piece specifies the CIFAR100 Train split dataset downloaded to `./data/` with `full` augmentation (random crop and horizontal flip) under the name `train`
```json
{
    "dataset": {
        // general pattern of specifying datasets
        // "internal-name-of-the-dataset": CLASS_SPEC,

        // `train` dataset specification, see the class for parameter docs
        "train": {
            "cls": "<class 'markovdwp.source.datasets.cifar.CIFAR100_Train'>",
            "root": "./data/",
            "augmentation": "full",
            "train_size": null,
            "random_state": null
        },
        ...
    }
}
```

#### feeds

Like `dataset`, `feeds` is also a dict of class-specs, except that each key **must** correspond to a dataset in `datasets`.

The following specifies a dataloader (also called `feed`) with batch size 256 and randomly pre-shuffled for the `train` dataset under the same name:
```json
{
    "feeds": {
        // general pattern of specifying feed 
        "<name-in-datasets>": CLASS_SPEC,

        "train": {
            "batch_size": 256,
            "shuffle": true,
            "pin_memory": true,
            "num_workers": 6
        }
    }
}
```

#### trainer

`trainer` is a dictionary of parameters, which specify the higher level parameters of the trainer `pytorch_lightning.Trainer`:

```json
{
    "trainer": {
        "max_epochs": 300,
        "track_grad_norm": 1.0,
        "val_check_interval": 1.0,
        "resume_from_checkpoint": null
    }
}
```

#### options

`options` is a dictionary of finer parameters of the experiment, such as specification of the applied method, the learning rate or coefficients in the regularization terms of the objective function:

```json
{
    "options": {
        "lr": 0.001,
            "coef": {
                "task.nll": 1.0,
                "model.l2_norm.features": 0.0001,
                "model.l2_norm.classifier": 0.0
            }
        }
    }
}
```

### Additional fields, specifc to `experiments/vae.py`

#### resampling (optional)
`resampling` the resampling strategy applied to the dataset, which is applied after the datasets are retrieved, but before the batch feeds are created. As such it facilitates renaming of the datasets and defining different subsamples thereof. The following example undersamples the `train` dataset and stores it under the same name, then independently creates its copy of under a new name `train_copy` (before undersampling), and creates a `reference` dataset, that is a deterministic random subset of the `test`.

```json
{
    "resampling": {
        "train": {
            "source": "train",
            "kind": "undersample",
            "size": null,
            "random_state": null
        },
        "train_copy": "train",
        "reference": {
            "source": "test",
            "kind": "subset",
            "size": 256,
            "random_state": 1381736037
        },
        ...
    }
}
```

#### runtime
`runtime` this is a partial class-spec, that specifies the training paradigm of the Vartiational Autoencoder. May specify any class, but only the following are implemented:
* `"<class 'markovdwp.runtime.vae.SGVBRuntime'>"` SGVB method of [Kingma and Welling (2014)](https://openreview.net/forum?id=33X9fd2-9FyZd)
* `"<class 'markovdwp.runtime.vae.IWAERuntime'>"` IWAE method of [Burda et al. (2016)](https://openreview.net/forum?id=RFZ6gFAik9K)
* `"<class 'markovdwp.runtime.vae.TRIPRuntime'>"` Maxlmum Likelihood for
Tensor Ring Induced Prior of [Kuznetsov et al. (2019)](http://papers.nips.cc/paper/8664-a-prior-of-a-googol-gaussians-a-tensor-ring-induced-prior-for-generative-models.pdf)

#### vae
`vae` is the field used in place of `model` in the VAE experiemnts. Has the following structure:
```json
{
    "vae": {
        "encoder": CLASS_SPEC,
        "decoder": CLASS_SPEC,
        "options": {...}
    }
}
```
`encoder` and `decoder` fields contain the class specs of the Encoder and the Decoder, respectively. `options` field allows non-redundant specification of shared parameters for both models. 


## Obtaining source kernel datasets

Source kernel dataset is built from the parameters of many convolutional networks trained on a common image dataset, e.g. `CIFAR100`.

To train a **single model** on `gpu-0` it is necessary to run the following line from the root of the installed package:
```bash
CUDA_VISIBLE_DEVICES=0 python -m markovdwp.source \
    ./experiments/configs/cifar100-source.json \
    --target ./data/single__cifar100-model.gz \
    --tag cifar100-one-model
```

The file `./experiments/configs/cifar100-source.json` contains the manifest of the source kernel acquisition experiment, the resulting trained model of which is saved under the name `./data/single__cifar100-model.gz`.

### Source models

In order to get the complete source kernel dataset `./data/kernels__cifar100-models` we need
to train a lot of CNN. For this the following command suffices:
```bash
mkdir -p ./data/cifar100-models

# Train 2 models on each device (see doc in `./scripts/source.sh`)
./scripts/source.sh 2 ./experiments/configs/cifar100-source.json \
    ./data/cifar100-models "cifar100_model" 0 1 2 3
```

This creates a folder `./data/cifar100-models`, which the trained models will be saved into. Then the `source.sh` script is run which performs the following steps:
1. Spawns **parallel `tmux` sessions, forked form the current bash environment** under unique names, that retain environment variable settings.
2. **Each session is assigned its gpu id** from the list. If the GPU id list has repeated ids, then a new session is forked and gets assigned the id. This allows to run different experiments on the same device **in isolation** for better resource utilization. No load balancing is performed other than this.
3. Within each session 2 **experiments are run, one after another**, with the specified experiment **manifest** and saving results into the specified folder `./data/cifar100-models` under random unique names starting with `cifar100_model` prefix.

### Source Kernel Datasets

Kernel Datasets are special subclasses of `torch.utils.data.Dataset`, that allow seamless access to the convolution datasets collected from the trained models. However, before these object can be used, it is necessary to collect the kenrels form the trained models in a suitable format. After a sample of models has been trained the source kernel dataset is collected with the following command:
```bash
# see documentation in `dataset.sh`
./scripts/dataset.sh ./data/cifar100-models

# python -m markovdwp.source.kernel ./data/cifar100-models
```

This automatically creates a folder next to `./data/cifar100-models` with the name `./data/kernels__cifar100-models` (just adds the `kernels__` prefix to the base name). In this new folder the script creates `meta.json` and a bunch of binary files, storing the trained weights of each convolutional layer pooled from all models in a consistent order.

The JSON file contains important meta information on the collected dataset:
* the shape, data type and storage of convolutions from a specific layer of the common model's architecture
* paths to snapshots of the trained models in exactly the same order as the convolution kernels in each layer
* the experiment manifest used to train all models

The following piece specifies a kernel dataset of a **7x7** convolution layer `features.conv0` (key) with **3 input** and **128 output** channels collected from **100 models** (`shape` field). The tensor of type **float32** (`dtype` field) is stored in flat file format in *opaquely named binary file* **vqbxr3rlb.bin** (`vault` field).
```json
{
  "dataset": {
    "features.conv0": {
      "shape": [100, 128, 3, 7, 7],
      "dtype": "torch.float32",
      "vault": "vqbxr3rlb.bin"
    }
  }
}
```

Two kinds of source kernel dataset objects are implemented: `SingleKernelDataset` and `MultiKernelDataset`. Each dataset returns a slice of a convolution kernel and a consistent label associated with the layer, where it came from.

* `SingleKernelDataset` -- a dataset of kernels from a single layer, any convolution kernel slicing is supported: 'm' by model, 'i' by inputs, 'o' outputs and combinations thereof.
* `MultiKernelDataset` -- a dataset of kernels from several, possible incompatible layers, that pads the spatial dimensions of smaller layers to the common shape. Only spatial slices are supported.

The objects can be used like this:
```python
from markovdwp.source import SingleKernelDataset, MultiKernelDataset

# ream and validate the kenel dataset
info = SingleKernelDataset.info('./data/kernels__cifar100-models')
print(info.keys())
# >>> dict_keys(['features.conv0', 'features.conv1', 'features.conv2', 'features.conv3'])

dataset = SingleKernelDataset('./data/kernels__cifar100-models', 'features.conv2')
# SingleKernelDataset(
#   source=`features.conv2`
#   kernel=(5, 5)
#   dim=(0, 1, 2)
#   n_models=100
#   root="data/kernels__cifar100-models"
# )

dataset = MultiKernelDataset('./data/kernels__cifar100-models', [
    'features.conv0',
    'features.conv2'
])
# MultiKernelDataset(
#   source=`('features.conv0', 'features.conv2')`
#   kernel=(7, 7)
#   n_models=100
#   root="data/kernels__cifar100-models"
# )
```

## Training Variational Autoencoders for the Deep Weight Prior

The original DWP requires as one Variational Autoencoder per implicit prior, hence one per layer. In order to train autoencoders separately the following commands are required:
```bash
# run `python ./experiments/vae.py --help` for documentation
CUDA_VISIBLE_DEVICES=0 python ./experiments/vae.py \
    --manifest ./experiments/configs/vae7x7_conv0.json \
    --target ./experiments/vae7x7_conv0.gz

CUDA_VISIBLE_DEVICES=0 python ./experiments/vae.py \
    --manifest ./experiments/configs/vae5x5_conv1.json \
    --target ./experiments/vae5x5_conv1.gz

CUDA_VISIBLE_DEVICES=0 python ./experiments/vae.py \
    --manifest ./experiments/configs/vae5x5_conv2.json \
    --target ./experiments/vae5x5_conv2.gz

CUDA_VISIBLE_DEVICES=0 python ./experiments/vae.py \
    --manifest ./experiments/configs/vae5x5_conv3.json \
    --target ./experiments/vae5x5_conv3.gz
```
Upon completion this creates four model snapshots in `./experiments/`. For parallel training it is better to run these in parallel detached `tmux` sessions from the root of the package.

To train a VAE on pooled kernel dataset of convolution form layers 1, 2, and 3 of `markovdwp.models.cifar.SourceCIFARNet` the following config should be used:
```bash
python ./experiments/vae.py \
    --manifest ./experiments/configs/vae5x5_pooled.json \
    --target ./experiments/vae5x5_pooled.gz
```

## Training models on new datasets with DWP

Finally after the kernel datasets have been collected and the VAEs trained it becomes posible to train new Bayesian networks on new datasets with Deep Weight Priors. The following command run a single experiment `./experiments/configs/dwp.json` and stores the trained
Bayesian network in `./data/dwp/`, but not the VAEs of the DWP.
```bash
# run `python ./experiments/dwp.py --help` for documentation
python ./experiments/dwp.py --manifest "./experiments/configs/dwp.json" --target "./data/dwp/"  --gpus 0

```
Optional arguments to this experiment include:
* `--priors` overrides training setting of the DWP encoders specified in the manifest (decoders are never retrained)
  * `trainable` encoders are trained along with the main variation approximation
  * `fixed` encoders are not trained and kept with their pre-trained parameters
  * `collapsed` encoders in the VAEs are replaced by the latent variable's prior

* `--init` override the parameter initialization specified in the manifest
  * `default` use torch's default random init (layer specific)
  * `prior` sample kernels from the VAE associated with thE DWP
  * `<path>` the path to kernel dataset, which to sample kernels from

The grid search version of this experiment uses (wandb sweeps)[https://docs.wandb.com/sweeps] which facilitate and greatly simplify massive parallel experimentation. The general procedure is to **create a sweep from a YAML sweep spec**, and then **spawn wandb agents** that communicate and fetch jobs from the wandb's sweep job server.
The command:
```bash
wandb sweep ./experiments/sweeps/dwp.yaml
```
returns an agent spawner command that looks something like this
```bash
wandb agent "<username>/DWP Grid Search Machine/z1o6k6hq"
```
Running this command launches a single agent, but without any control of what device it runs on. This issue can be solved by using the `sweep.sh` script, which automatically spawns a single isolated wandb agent on each id from the specified list of GPU ids. The following launches 3 agents per GPU, that are properly isolated from each other and maintain own GPU device and PRNG context.
```bash
./scripts/sweep.sh "<username>/DWP Slice Replication Machine TWO/z1o6k6hq" 0 0 0 1 1 1 2 2 2 3 3 3
```

Sweep specs for grid search experiments on Bayesian deep networks included in this package are:
* `dwp.yaml` -- use DWP and the auxiliary ELBO proposed in [Atanov et al. (2019) eq. (7)](https://openreview.net/forum?id=ByGuynAct7)
* `classic.yaml` -- uses Standard Gaussian prior on the weight's variational approximation, instead of the DWP
* `sparsevd.yaml` -- uses log-uniform prior for Sparse Variational Dropout of [Molchanov et. al (2017) eq. (8)](http://proceedings.mlr.press/v70/molchanov17a.html) on the variational approximation, rather than the DWP

Each sweep specifies the base experiment manifest and the necessary modifications to it that fulfil the desired experiment settings, and defines the hyperparameter grid:
* `init` -- specify the initialization of the kernels of the Bayesian Network
* `priors` -- specify how the encoders in the DWP are treated
* `dataset__train__train_size` -- determines the size of the sample used for training
* `order` -- a dummy variable that allows replicating the same experiment as many times as necessary

Sweeps are distinct in the following parameters:
* `model__cls` -- the class of the model used for training `markovdwp.models.cifar.BayesCIFARNetVD` or `markovdwp.models.cifar.BayesCIFARNetGaussian`
* `options__kind` -- the training method
  * `implicit` -- use the auxiliary ELBO proposed by [Atanov et al. (2019)](https://openreview.net/forum?id=ByGuynAct7)
  * `classic` -- use the common ELBO proposed by [Kingma and Welling (2014)](https://openreview.net/forum?id=33X9fd2-9FyZd)


# References

.. [1] Atanov, A. and Ashukha, A. and Struminsky, K. and Vetrov, D. and Welling, M. (2019). The Deep Weight Prior. International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=ByGuynAct7

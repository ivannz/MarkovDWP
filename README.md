# Markov Deep Weight Prior

The reviews of the Deep Weight Prior, proposed by [Atanov et al. (2019)](https://openreview.net/forum?id=ByGuynAct7), were mostly concerned with modelling issues: **factorization** and **layer and channel independence**, since there is ample evidence regarding the dependence of convolutional kernel slices within layers and between them, e.g. [Olah et al.(2020)](https://distill.pub/2020/circuits/zoom-in).

In this project we set out to find a solution, a hierarchical model, that would enable generation of interdependent slices for the filter bank of a single convolutional layer. Having achieved a substantial result, we could explore further tentative directions: generative model for entire layers, progressive growth of neural networks in architecture search.


## Developers

Please install the cloned repo in editable mode for easier deployment:

```bash
git clone https://github.com/ivannz/MarkovDWP.git
cd MarkovDWP
pip install -e .
```

The base stack will be `python>=3.8`, `torch>=1.4` and `pytorch-lighting` as the framework.

The repo has the following structure:
* `markovdwp` -- the package with method implementations and the core experimentation backend.
* `experiments` -- the directory where experiments will be designed and their prototypes kept.


# References

.. [1] Atanov, A. and Ashukha, A. and Struminsky, K. and Vetrov, D. and Welling, M. (2019). The Deep Weight Prior. International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=ByGuynAct7

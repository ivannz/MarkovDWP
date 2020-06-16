# Markov Deep Weight Prior

The reviews of the Deep Weight Prior, proposed by [Atanov et al. (2019)](https://openreview.net/forum?id=ByGuynAct7), were mostly concerned with modelling issues: **factorization** and **layer and channel independence**, since there is ample evidence in regarding the dependence of convolutional kernel slices within layers and between them, e.g. [Olah et al.(2020)](https://distill.pub/2020/circuits/zoom-in).

In this project we set out to find a solution, a hierarchical model, that would enable generation of interdependent slices for the filter bank of a single convolutional layer. Having achieved a substantial result, we could explore further tentative deirections: generative model for entire layers, progressive growth of neural networks in architecture search.


# References

.. [1] Atanov, A. and Ashukha, A. and Struminsky, K. and Vetrov, D. and Welling, M. (2019). The Deep Weight Prior. International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=ByGuynAct7

import gzip

import tqdm
import wandb
import torch

import matplotlib.pyplot as plt

from markovdwp.priors import ImplicitSlicePrior
from markovdwp.priors.implicit import to_q

from markovdwp.source.utils.plotting import plot_slices
from markovdwp.source.utils.runtime import get_instance
from markovdwp.source.utils.common import weighted_sum
from markovdwp.source.base import linear, partial

from markovdwp.nn import named_penalties

from cplxmodule.nn.relevance.real import Conv2dVD


def unpack(state):
    module = get_instance(**state['model'])
    module.load_state_dict(state['state'])
    return module


def main(kind='implicit', n_draws=1, max_epochs=250, collapsed=True, **kwargs):
    assert kind in ('implicit', 'classic')

    # use zero-th device by default
    device_ = torch.device('cuda:0')

    # open the vae and create the Implicit Prior
    target = '/home/ivan.nazarov/Github/MarkovDWP/experiments/dwp-vae__kqqz3mpl.gz'
    with gzip.open(target, 'rb') as fin:
        snapshot = torch.load(fin, map_location=torch.device('cpu'))

    decoder = unpack(snapshot['decoder']).requires_grad_(False)
    encoder = unpack(snapshot['encoder']).requires_grad_(False)
    if collapsed:
        encoder = encoder.z_dim, 1, 1

    ip7x7 = ImplicitSlicePrior(decoder, encoder).to(device_)

    # create a simple conv layer
    conv = Conv2dVD(3, 32, 7).to(device_)

    # Adam with linear lr schedule
    optim = torch.optim.Adam(conv.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, partial(linear, t0=0, t1=max_epochs))

    # pre-create penalties
    penalties = {}
    if kind == 'implicit':
        penalties = {
            'kl_div': partial(ip7x7.penalty, n_draws=n_draws, coef=1.)
        }

    # train loop
    for i in tqdm.trange(max_epochs):
        optim.zero_grad()

        # dummy task
        X = torch.randn(1, 3, 32, 32).to(device_)
        task = {'task': 0. * (conv(X)).sum()}
        kl_div = dict(named_penalties(conv, penalties=penalties,
                                      prefix='kl_div', reduction='sum'))
        value, terms = weighted_sum({**kl_div, **task}, **{'': 1.})

        # backprop
        value.backward()
        optim.step()

        sched.step()

        # report
        if (i % 5) == 0:
            wandb.log({
                'slices': plot_slices(
                    conv.weight.detach().cpu().flatten(0, 1),
                    f_aspect=3 / 2, cmap=plt.cm.coolwarm)
            }, commit=False)

        # get entropy
        with torch.no_grad():
            q, = to_q(conv)

        wandb.log({
            'model.entropy': q.entropy().mean(),
            **terms
        }, commit=True)


wandb.init(project='Implicit Prior Diversity')
wandb.config.setdefaults({
    'kind': 'implicit',
    'n_draws': 25,
    'max_epochs': 250,
    'collapsed': True
})

main(**wandb.config)

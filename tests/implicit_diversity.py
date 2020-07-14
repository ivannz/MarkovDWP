import gzip

import tqdm
import wandb
import torch

import matplotlib.pyplot as plt

from funvtools import partial

from markovdwp.priors import ImplicitSlicePrior
from markovdwp.priors.implicit import to_q

from markovdwp.source.utils.plotting import plot_slices
from markovdwp.source.utils.common import weighted_sum, linear

from markovdwp.utils.runtime import get_instance

from markovdwp.nn import named_penalties

from cplxmodule.nn.relevance.real import Conv2dVD


def unpack(state):
    module = get_instance(**state['model'])
    module.load_state_dict(state['state'])
    return module


def main(elbo_kind='implicit', encoder_kind='collapsed',
         max_epochs=250, use_sched=True, n_draws_q=1, n_draws_r=1,
         verbose=False, **kwargs):
    assert elbo_kind in ('implicit', 'classic')

    assert encoder_kind in ('collapsed', 'fixed', 'trainable')

    # use zero-th device by default
    device_ = torch.device('cuda:0')

    # open the vae and create the Implicit Prior
    target = '/home/ivan.nazarov/Github/MarkovDWP/experiments/dwp-vae__kqqz3mpl.gz'
    with gzip.open(target, 'rb') as fin:
        snapshot = torch.load(fin, map_location=torch.device('cpu'))

    decoder = unpack(snapshot['decoder']).requires_grad_(False)
    encoder = unpack(snapshot['encoder'])

    encoder.requires_grad_(encoder_kind == 'trainable')
    if encoder_kind == 'collapsed':
        encoder = encoder.event_shape

    # create a simple conv layer
    conv = Conv2dVD(3, 32, 7)

    # attach the implict prior to it, so that we can keep them together
    ip7x7 = conv.ip7x7 = ImplicitSlicePrior(decoder, encoder)

    conv.to(device_)

    # Adam with linear lr schedule
    optim, sched = torch.optim.Adam(conv.parameters(), lr=1e-3), None
    if use_sched:
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, partial(linear, t0=0, t1=max_epochs))

    # pre-create penalties
    penalties = {}
    if elbo_kind == 'implicit':
        penalties = {
            'kl_div': partial(ip7x7.penalty, n_draws_q=n_draws_q,
                              n_draws_r=n_draws_r, coef=1.)
        }

    # train loop
    for i in tqdm.trange(max_epochs, disable=not verbose):
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
        if sched is not None:
            sched.step()

        # report
        if (i % 5) == 0 or (i+1 == max_epochs):
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
    'elbo_kind': 'implicit',
    'encoder_kind': 'collapsed',
    'max_epochs': 250,
    'use_sched': True,
    'n_draws_q': 1,
    'n_draws_r': 1,
})

main(**wandb.config)

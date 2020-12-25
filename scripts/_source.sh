#!/bin/bash
#
# Train a number of source models.
#
# $ ./source.sh `N` `manifset` `target` `--gpus 3` `--tag cifar100`
#
# does the folowing
#  repeat `N` times:
#     train some model on some dataset (all specified in `manifest`)
#     store the model's trained weights into `target` under a unique name
#
#  see `python -m markovdwp.source --help` for other details.

n_repl="${1}"; shift
manifest="${1}"; shift
target="${1}"; shift

for ((i=1;i<=n_repl;i++)); do
    # tag="run-$(printf '%02d' ${i})"

    # `ampersand` forks into a parallel bash process
    # redirect stdout to the individual file
    python -m markovdwp.source "${manifest}" --target "${target}" $@ \
        --seed deterministic > /dev/null
done

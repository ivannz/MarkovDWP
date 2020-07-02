#!/bin/bash
#
# Runs wandb agents on separate devices
#
# $ ./sweeper.sh `sweep-id` `gpu` `[* gpu]`
#
#  Launches a new agent in parallel bash subprocesses
#   for every device in `gpus`
#
#  see `wandb sweep --help` for details on creating sweeps
#  see `wandb agent --help` for details on launching agents

sweep="${1}"; shift

for device in "$@"; do
    # `ampersand` forks into a parallel bash process
    CUDA_VISIBLE_DEVICES=${device} wandb agent "${sweep}" > /dev/null &

    # wait for 2-7 seconds
    sleep $((2 + RANDOM % 5));
done

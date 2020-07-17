#!/bin/bash
#
# Runs wandb agents on separate devices in detached tmux-sessions
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
    # for better random names see <https://gist.github.com/earthgecko/3089509>
    name=$(hexdump -n 16 -v -e '/1 "%02X"' /dev/urandom)

    # w/o any commands tmux starts a new session with `bash` and which resets
    # conda env to base. Instead we just run env folllowed by python within
    # the environment, from where tmux was launched.
    tmux new-session -d -s "${name}" \
        env CUDA_VISIBLE_DEVICES=${device} \
            wandb agent "${sweep}"
    # for debug: python -c"import torch; print(torch.cuda.device_count())"

    # wait for 2-7 seconds
    sleep $((2 + RANDOM % 5));
done

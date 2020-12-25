#!/bin/bash
#
# Train some number of source models on each device.
#
# $ ./source.sh `N` `manifset` `target` `tag` `gpu` `[* gpu]`
#
#  Launches source dataset acquisition scripts in parallel bash
#   subprocesses one per device in `gpus`
#
#  see `./_source.sh` for details

n_repl="${1}"; shift
manifest="${1}"; shift
target="${1}"; shift
tag="${1}"; shift

[ ! -d "${target}" ] && mkdir -p "${target}"

for device in "$@"; do
    # for better random names see <https://gist.github.com/earthgecko/3089509>
    name=$(hexdump -n 16 -v -e '/1 "%02X"' /dev/urandom)

    # w/o any commands tmux starts a new session with `bash` and which resets
    # conda env to base. Instead we just run env folllowed by python within
    # the environment, from where tmux was launched.
    tmux new-session -d -s "${name}" \
        env CUDA_VISIBLE_DEVICES=${device} \
            ./_source.sh ${n_repl} "${manifest}" "${target}" --gpus 0 --tag "${tag}"

    # wait for 2-7 seconds
    sleep $((2 + RANDOM % 5));
done

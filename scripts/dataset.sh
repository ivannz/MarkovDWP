#!/bin/bash
#
# Process trained source models located in `root` into kernel dataset.
#
# Creates a subdirectory next to `root` under a name prefixed with `kernels__`.
# The subdirectory contains metadata with the specs of the kernels and the
# config of the model used to generate them. Also contains binary files holding
# the raw data of the tensors.
#
# $ ./dataset.sh `root` [--tag "optional_suffix"]
#

python -m markovdwp.source.dataset $@

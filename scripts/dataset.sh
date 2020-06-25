#!/bin/bash
#
# Process trained source models located in `root` into kernel dataset, then
# create binary filea and metadata in a subdirectory of `target` under a
# special name, determined by the class of the _trained_ model.
#
# $ ./dataset.sh `root` `target`
#

python -m markovdwp.source.dataset $@

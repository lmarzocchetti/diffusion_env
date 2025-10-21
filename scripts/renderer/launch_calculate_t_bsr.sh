#!/bin/sh

set -xe

python calculate_t_bsr.py \
    --renders_folder ../../out/renderer \
    --output_path ../../T_bsr.npz \
    --epsilon 5e-3
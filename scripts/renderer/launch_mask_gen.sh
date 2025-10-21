#!/bin/sh

set -xe

python generate_mask.py \
    --input_bsr_path ../../T_bsr.npz \
    --input_env_path ../../resources/meadow_2_90deg.exr \
    --input_stroke_path ../../resources/stroke.exr \
    --input_albedo_path ../../resources/albedo.exr \
    --mult_constant 1e-5 \
    --delta 0.0008 \
    # --resize_env \
    # --use_dense_matrix \
    # --color_render
#!/bin/sh

# PQ
# python test_inpainting_sd.py --model_path ./finetune_lora_sdxl_pq_100/checkpoint-5000 --model_version sdxl --test_input_dir ../test_input --color_space pq

# python test_inpainting_sd.py --model_path ./finetune_lora_sdxl_hlg2/checkpoint-5000 --model_version sdxl --test_input_dir ../test_input --color_space hlg

# PQ
# python test_inpainting_sd.py --model_path ./finetune_lora_pq/checkpoint-5000 --model_version sd2 --test_input_dir ../test_input --color_space pq
# Reinhard
# python test_inpainting_sd.py --model_path ./finetune_lora_no_resizing/checkpoint-5000 --model_version sd2 --test_input_dir ../test_input --color_space reinhard
# HLG
# python test_inpainting_sd.py --model_path ./finetune_lora_hlg/checkpoint-5000 --model_version sd2 --test_input_dir ../test_input --color_space hlg

# PQ Img2img
python test_inpainting_sd.py --model_path ./finetune_lora_pq/checkpoint-5000 --model_version img2img --test_input_dir ../test_input --color_space pq
# Reinhard Img2img
# python test_inpainting_sd.py --model_path ./finetune_lora_no_resizing/checkpoint-5000 --model_version img2img --test_input_dir ../test_input --color_space reinhard
# HLG Img2img
# python test_inpainting_sd.py --model_path ./finetune_lora_hlg/checkpoint-5000 --model_version img2img --test_input_dir ../test_input --color_space hlg

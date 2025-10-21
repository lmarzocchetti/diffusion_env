#!/bin/sh

# accelerate launch --mixed_precision="fp16" ./train_text_to_image_lora.py \
#     --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
#     --train_data_dir="/home/rhohen/Workspace/diffusion-env/data_precomputed_npy_pq" \
#     --caption_column="additional_feature" \
#     --mixed_precision="fp16" \
#     --resolution=768 \
#     --train_batch_size=1 \
#     --max_train_steps=5000 \
#     --image_column="file_name" \
#     --learning_rate=1e-04 \
#     --max_grad_norm=1 \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps=0 \
#     --output_dir="./finetune_lora_pq/" \
#     --checkpointing_steps=500 \
#     --validation_prompt="Equirectangular environment map" \
#     --seed=1337
#     # --gradient_accumulation_steps=2 \
#     # --dataloader_num_workers=0 \


# SDXL
accelerate launch --mixed_precision="fp16" ./train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir="/home/rhohen/Workspace/diffusion-env/data_precomputed_npy_pq" \
    --caption_column="additional_feature" \
    --mixed_precision="fp16" \
    --resolution=768 \
    --train_batch_size=1 \
    --max_train_steps=5000 \
    --image_column="file_name" \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="./finetune_lora_sdxl_pq_100/" \
    --checkpointing_steps=500 \
    --validation_prompt="Equirectangular environment map" \
    --seed=1337
    # --gradient_accumulation_steps=2 \
    # --dataloader_num_workers=0 \

# DiffusionEnv
Stable Diffusion pipeline to moving lights in HDR images (Environment Maps)

### Explanation (in short)
Stable diffusion models are trained with LDR images. Obviously we can't feed our HDR images into this model directly. So there is the needs to adapt those weights to "support" HDR images as imputs. So i have created a custom Dataset of HDR images (outdoor images), and attached a LoRA to a Stable Diffusion 2 inpainting pipeline, and trained with these HDR images (more details in the thesis, using some color spaces like PQ, HLG, ecc). So the user need to provide 2 masks to say where the new and the old light are and the input image with a simple editing (with prompt, or using the sample prompt). So the output is the new inpainted image, hopefully preserving color and intensity of the old light.

### Images
Original Image: <img width="1024" height="512" alt="CapeHill_Original" src="https://github.com/user-attachments/assets/5d8d70e4-2815-4868-b345-1c387b1296d9" />

Input Image (edited): <img width="1024" height="512" alt="CapeHill_Input" src="https://github.com/user-attachments/assets/1b10b4dc-3bdb-4c30-9e12-9b895af0c888" />

Mask 1: <img width="1024" height="512" alt="CapeHill_mask_1_sky_user" src="https://github.com/user-attachments/assets/120124c8-2650-42a7-8843-df75268f8257" />

Mask 2: <img width="1024" height="512" alt="CapeHill_mask_2_light_user" src="https://github.com/user-attachments/assets/0bc61061-1b8b-4fbe-9772-5a203084c13a" />

Output Image: <img width="1024" height="512" alt="CapeHill_Output" src="https://github.com/user-attachments/assets/879f0df7-e208-4269-a83d-bf99fc27dba3" />

Rendering with the original image: <br>
<img width="512" height="256" alt="CapeHill_Render_Original" src="https://github.com/user-attachments/assets/114a9c63-f53c-45b0-8d35-13d5d78efdad" />

Rendering with the output image: <br>
<img width="512" height="256" alt="CapeHill_Render_Inpaint" src="https://github.com/user-attachments/assets/3c4816dd-c30c-4117-b283-fbde79096340" />


### Usage
This is a work in progress because i need to retrieve the weights of the trained LoRA adapters and/or retreive my custom dataset. Both of these are in a separate machine (school computer, which my professor own, so...)

But with a custom dataset you can look at the scripts in order:
```
$ scripts/prepare_data_hdri.py
$ scripts/generate_metadata_jsonl.py
$ scripts/launch_train_lora.sh
```

which are for training the LoRA adapter.

Then 
```
$ scripts/precompute.py
```
is to generate the .npy files to using the renderer baked in the notebook. Or you can use simply Blender to see a 3d scene rendered with the new output image

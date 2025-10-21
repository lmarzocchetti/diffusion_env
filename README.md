# DiffusionEnv
Stable Diffusion pipeline to moving lights in HDR images (Environment Maps)

### Explanation (in short)
Stable diffusion models are trained with LDR images. Obviously we can't feed our HDR images into this model directly. So there is the needs to adapt those weights to "support" HDR images as imputs. So i have created a custom Dataset of HDR images (outdoor images), and attached a LoRA to a Stable Diffusion 2 inpainting pipeline, and trained with these HDR images (more details in the thesis, using some color spaces like PQ, HLG, ecc). So the user need to provide 2 masks to say where the new and the old light are and the input image with a simple editing (with prompt, or using the sample prompt). So the output is the new inpainted image, hopefully preserving color and intensity of the old light.

### Images
Original Image:

Input Image (edited):

Mask 1:

Mask 2:

Output Image:

Rendering with the original image:

Rendering with the output image:

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
from pathlib import Path
from PIL import Image

import OpenEXR
import Imath
import numpy as np
import cv2

def load_exr(filename):
    """Load an EXR image and return it as a NumPy array."""
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header['channels'].keys())  # Convert to a list
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read channels
    img = {ch: np.frombuffer(exr_file.channel(ch, pixel_type), dtype=np.float32)
           .reshape(height, width) for ch in channels}

    return img, width, height, channels

def save_exr(filename, img, channels):
    """Save an EXR image from a NumPy dictionary."""
    channels = list(channels)  # Ensure channels is a list
    height, width = img[channels[0]].shape

    header = OpenEXR.Header(width, height)
    exr_file = OpenEXR.OutputFile(filename, header)

    # Convert channels to byte format
    channel_data = {ch: img[ch].astype(np.float32).tobytes() for ch in channels}
    exr_file.writePixels(channel_data)

def resize_exr(img, new_width, new_height):
    """Resize an EXR image using OpenCV."""
    resized_img = {ch: cv2.resize(img[ch], (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
                   for ch in img}
    return resized_img

def luminance(rgb):
    return rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)

def change_luminance(rgb: np.ndarray, l_out: np.ndarray):
    l_in: np.ndarray = luminance(rgb)
    result_div = None
    with np.errstate(divide='ignore', invalid='ignore'):
        result_div = l_out / l_in
        result_div[np.isinf(result_div)] = 0
        result_div[np.isnan(result_div)] = 0
    return rgb * np.expand_dims(result_div, axis=-1)
    # return rgb * np.expand_dims((np.where(l_in != 0, l_out / l_in, 0)), axis = -1) # FIXME: Ho aggiunto where

def rgb_to_reinhard(rgb, max_white_l=100):
    l_old = luminance(rgb)
    numerator = l_old * (1.0 + (l_old / (max_white_l * max_white_l)))
    l_new = numerator / (1.0 + l_old)

    return change_luminance(rgb, l_new)

def rgb_to_hlg(rgb):
    hlg = None
    with np.errstate(divide='ignore', invalid='ignore'):
        hlg = np.where(rgb <= 1.0,
                    0.5 * np.sqrt(rgb),
                    0.17883277 * np.log(rgb - 0.28466892) + 0.55991073)
        hlg[np.isinf(hlg)] = 0
        hlg[np.isnan(hlg)] = 0
    return hlg


def hlg_to_rgb(hlg):
    rgb = None
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.where(hlg <= 0.5,
                    np.square(2.0 * hlg),
                    np.exp((hlg - 0.55991073) / 0.17883277) + 0.28466892)
        rgb[np.isinf(rgb)] = 0
        rgb[np.isnan(rgb)] = 0
    return rgb

def pq_to_rgb(rgb: np.ndarray, mul: int = 10000) -> np.ndarray:
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ret = None
    with np.errstate(divide='ignore', invalid='ignore'):
        E_p = rgb ** (1/m2)
        par = np.maximum((E_p - c1), 0) / (c2 - c3 * E_p)
        ret = mul * (par ** (1/m1))
        ret[np.isinf(ret)] = 0
        ret[np.isnan(ret)] = 0
    
    return np.clip(ret, 0, 1)

def rgb_to_pq(pq: np.ndarray, div: int = 10000) -> np.ndarray:
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ret = None
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = pq / div
        ret = ((c1 + c2 * (Y ** m1)) / (1 + c3 * (Y ** m1))) ** m2
        ret[np.isinf(ret)] = 0
        ret[np.isnan(ret)] = 0

    return ret

def main():
    input_dir = "../data/"
    output_dir = "../data_precomputed_npy_pq/"
    # output_dir = "../data_prova/"
    scale_factor = 0.5
    
    folder_path = Path(input_dir)
    
    files = [f.name for f in folder_path.iterdir() if f.is_file()]
        
    for file in files:
        input_exr = f"{input_dir}{file}"
        img, width, height, channels = load_exr(input_exr)
        # new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        
        # RESIZE MODE NPY
        # new_width, new_height = 768, 768 # Input of stable diffusion 2
        # resized_img = resize_exr(img, new_width, new_height)
        # resized_rgb_image = np.stack([resized_img['R'], resized_img['G'], resized_img['B']], axis=-1)
        # rein_out = rgb_to_reinhard(resized_rgb_image)
        # np.save(f"{output_dir}/{file[0:-4]}.npy", rein_out)
        
        # NORMAL MODE NPY
        img_arr = np.stack([img['R'], img['G'], img['B']], axis=-1)
        # rein_out = rgb_to_reinhard(img_arr)
        # rein_out = rgb_to_hlg(img_arr)
        rein_out = rgb_to_pq(img_arr, div=100)
        np.save(f"{output_dir}/{file[0:-4]}.npy", rein_out)
        
        # SAVE PNG MODE
        # img = Image.fromarray((rein_out * 255).clip(0, 255).astype(np.uint8))
        # img.save(f"{output_dir}/{file[0:-4]}.png")
    
if __name__ == "__main__":
    main()
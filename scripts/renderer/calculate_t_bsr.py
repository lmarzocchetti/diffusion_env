from argparse import ArgumentParser

import numpy as np
import OpenEXR
import pywt

from Imath import PixelType
from scipy.sparse import bsr_array, save_npz

def load_exr(filename):
  exr = OpenEXR.InputFile(filename)
  dw = exr.header()['dataWindow']
  width = dw.max.x - dw.min.x + 1
  height = dw.max.y - dw.min.y + 1

  img = np.zeros((height, width, 3), dtype=np.float32)
  for i, c in enumerate('RGB'):
    buffer = exr.channel(c, PixelType(OpenEXR.FLOAT))
    img[:, :, i] = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)

  exr.close()
  return img

def save_exr(filename, img):
  height, width, _ = img.shape
  header = OpenEXR.Header(width, height)
  exr = OpenEXR.OutputFile(filename, header)
  
  r, g, b = np.split(img, 3, axis=-1)
  exr.writePixels({'R': r.tobytes(),
                   'G': g.tobytes(),
                   'B': b.tobytes()})
  exr.close()

def create_parser() -> ArgumentParser:
    arg_parser: ArgumentParser = ArgumentParser("Calculate T_bsr")
    arg_parser.add_argument(
        "--renders_folder",
        type=str,
        required=True,
        help="Path to the folder containing all renders done with precompute.py script"
    )
    arg_parser.add_argument(
        "--output_path",
        type=str,
        default="./T_bsr.npz",
        help="Path to the output file"
    )
    arg_parser.add_argument(
        "--color_render",
        action="store_true",
        help="If the render in exr is done in RGB or greyscale"
    )
    arg_parser.add_argument(
        "--img_width",
        type=int,
        default=512,
        help="Width of the image"
    )
    arg_parser.add_argument(
        "--img_height",
        type=int,
        default=256,
        help="Height of the image"
    )
    arg_parser.add_argument(
        "--env_width",
        type=int,
        default=512,
        help="Width of the env map"
    )
    arg_parser.add_argument(
        "--env_height",
        type=int,
        default=256,
        help="Height of the env map"
    )
    arg_parser.add_argument(
        "--epsilon",
        type=float,
        default=5e-4,
        help="Epsilon for setting to 0 low values"
    )
    
    return arg_parser

def main():
    arg_parser: ArgumentParser = create_parser()
    args = arg_parser.parse_args()
    
    IMG_WIDTH, IMG_HEIGHT, ENV_WIDTH, ENV_HEIGHT = args.img_width, args.img_height, args.env_width, args.env_height
    
    T = np.zeros((IMG_WIDTH * IMG_HEIGHT, ENV_WIDTH * ENV_HEIGHT // 2), dtype=np.float32)
    for idx, (i, j) in enumerate(np.ndindex(ENV_HEIGHT // 2, ENV_WIDTH)):
        img = load_exr(f'{args.renders_folder}/{i:03d},{j:03}.exr')
        if args.color_render:
            coeffs = pywt.wavedec2(img, 'haar')
        else:
            coeffs = pywt.wavedec2(img[:, :, 0], 'haar')
        coeff_arr, _ = pywt.coeffs_to_array(coeffs)
        T[:, idx] = coeff_arr.ravel()
    
    T[np.isclose(T, 0, atol=args.epsilon)] = 0
    T_bsr = bsr_array(T, dtype=np.float32)
    save_npz(f'{args.output_path}', T_bsr)

if __name__ == "__main__":
    main()
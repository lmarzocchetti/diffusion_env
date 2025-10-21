from argparse import ArgumentParser

import numpy as np
import OpenEXR
import pywt
import cv2

from Imath import PixelType
from scipy.sparse import load_npz

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
    arg_parser: ArgumentParser = ArgumentParser("Generate mask")
    
    arg_parser.add_argument(
        "--input_bsr_path",
        type=str,
        required=True,
        help="Path to the T_bsr.npz file"
    )
    arg_parser.add_argument(
        "--input_env_path",
        type=str,
        required=True,
        help="Path to the env map file exr"
    )
    arg_parser.add_argument(
        "--input_stroke_path",
        type=str,
        required=True,
        help="Path to the stroke file exr"
    )
    arg_parser.add_argument(
        "--input_albedo_path",
        type=str,
        required=True,
        help="Path to the albedo file exr"
    )
    arg_parser.add_argument(
        "--output_mask_path",
        type=str,
        default="./mask.exr",
        help="Output path for the mask.exr file"
    )
    arg_parser.add_argument(
        "--output_env_path",
        type=str,
        default="./env.exr",
        help="Output path for the env.exr file"
    )
    arg_parser.add_argument(
        "--mult_constant",
        type=float,
        default=1e-4,
        help="Constant used to multiply the L"
    )
    arg_parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Constant used to multiply the L"
    )
    arg_parser.add_argument(
        "--color_render",
        action="store_true",
        help="If the render in exr is done in RGB or greyscale"
    )
    arg_parser.add_argument(
        "--use_dense_matrix",
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
        "--resize_env",
        action="store_true",
        help="If the env map should be resized to match the image size"
    )
    
    return arg_parser

def main():
    arg_parser: ArgumentParser = create_parser()
    args = arg_parser.parse_args()
    
    IMG_WIDTH, IMG_HEIGHT, ENV_WIDTH, ENV_HEIGHT = args.img_width, args.img_height, args.env_width, args.env_height

    T_bsr = load_npz(f"{args.input_bsr_path}")
    T = T_bsr.todense().astype(np.float32)

    L = load_exr(f"{args.input_env_path}")
    if args.resize_env:
        L = cv2.resize(L, (ENV_WIDTH, ENV_HEIGHT), interpolation=cv2.INTER_AREA)

    L = L[:ENV_HEIGHT // 2, :, :].reshape(-1, 3)

    if args.use_dense_matrix:
        B = T @ L * args.mult_constant
    else:
        # TODO: Controlla questo codice
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        if args.color_render:
            coeffs = pywt.wavedec2(img, "haar")
        else:
            coeffs = pywt.wavedec2(img[:, :, 0], "haar")
        _, coeff_slices = pywt.coeffs_to_array(coeffs)

        B = T_bsr @ L * args.mult_constant
        B = B.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        coeffs = pywt.array_to_coeffs(B, coeff_slices, output_format='wavedec2')
        B = pywt.waverec2(coeffs, 'haar', axes=(0, 1))
    
    stroke_img = load_exr(f'{args.input_stroke_path}').reshape(-1, 3)
    in_stroke = (1, 0, 0)
    out_stroke = (0, 1, 0)
    X_in = np.all(stroke_img == in_stroke, axis=-1)
    X_out = np.all(stroke_img == out_stroke, axis=-1)
    
    T_in = np.mean(T[X_in, :], axis=0)
    T_out = np.mean(T[X_out, :], axis=0)
    # TODO: Salvare T_in e T_out nell'altro script e caricarli qua
    
    delta = T_out[:, None] * L - T_in[:, None] * L
    
    p = load_exr(f"{args.input_albedo_path}").reshape(-1, 3)
    L_avg = np.mean(L, axis=0)
    p_avg = np.mean(p, axis=0)
    L_f = np.any(delta > args.delta * L_avg * p_avg, axis=-1)
    L_b = ~L_f # TODO: Cos'e' questo?
    
    env = L.copy()
    env[L_f, :] = (1, 0, 0)
    
    env_img = load_exr(f"{args.input_env_path}")
    if args.resize_env:
        env_img = cv2.resize(env_img, (ENV_WIDTH, ENV_HEIGHT), interpolation=cv2.INTER_AREA)
    env_img[:ENV_HEIGHT // 2, :, :] = env.reshape(ENV_HEIGHT // 2, ENV_WIDTH, 3)
    save_exr(f"{args.output_env_path}", env_img)
    
    mask = np.zeros_like(L)
    mask[L_f, :] = 1.0
    
    mask_img = np.zeros((ENV_HEIGHT, ENV_WIDTH, 3), dtype=np.float32)
    mask_img[:ENV_HEIGHT // 2, :, :] = mask.reshape(ENV_HEIGHT // 2, ENV_WIDTH, 3)
    
    save_exr(f"{args.output_mask_path}", mask_img)
    
if __name__ == "__main__":
    main()
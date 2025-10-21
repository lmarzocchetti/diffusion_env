import numpy as np
import OpenEXR

from Imath import Channel, PixelType
from shutil import move
from subprocess import run


ENV_WIDTH, ENV_HEIGHT = 512, 256


def save_exr(img, filename):
  height, width = img.shape
  header = OpenEXR.Header(width, height)
  header['channels'] = {'Y': Channel(PixelType(OpenEXR.FLOAT))}

  exr = OpenEXR.OutputFile(filename, header)
  exr.writePixels({'Y': img.tobytes()})
  exr.close()


if __name__ == "__main__":
  env = np.zeros((ENV_HEIGHT, ENV_WIDTH), dtype=np.float32)
  
  for i, j in np.ndindex(ENV_HEIGHT // 2, ENV_WIDTH):
    env[i, j] = 1.0
    save_exr(env, 'resources/env.exr')

    run(['blender', '-b', 'resources/test_scene.blend', '-P', 'scripts/env_loader.py', '-a'])

    move('resources/0001.exr', f'out/renderer/{i:03d},{j:03}.exr')
    env[i, j] = 0.0

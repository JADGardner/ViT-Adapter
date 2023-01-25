# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
import warnings
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import os
from contextlib import redirect_stdout
from io import StringIO
import tqdm

class NullIO(StringIO):
    def write(self, txt):
        pass

def silent(fn):
    """Decorator to silence functions."""
    def silent_fn(*args, **kwargs):
        with redirect_stdout(NullIO()):
            return fn(*args, **kwargs)
    return silent_fn

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('images', help='Image or Image Folder')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1.0,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file

    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=UserWarning)
      silent_init = silent(init_segmentor)
      print('Initializing segmentation model ...')
      model = silent_init(args.config, checkpoint=None, device=args.device)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
        
    # check if img_path is img or dir
    # if directory get paths to all files 
    # ending in .jpg or .png or .JPG or .PNG or .jpeg or .JPEG
    print(f'Segmenting images in {args.images}')
    imgs = {}
    if os.path.isdir(args.images):
      for file in os.listdir(args.images):
        if file.endswith('.jpg') or \
          file.endswith('.png') or \
          file.endswith('.JPG') or \
          file.endswith('.PNG') or \
          file.endswith('.jpeg') or \
          file.endswith('.JPEG'):
          imgs[file] = os.path.join(args.images, file)
    else:
      # get file name from img_path
      file = os.path.basename(args.images)
      imgs = {file: args.images}

    mmcv.mkdir_or_exist(args.out)

    # use tqdm to show progress bar
    for img_name, img_path in tqdm.tqdm(imgs.items()):
        # outpath as a .png instead of any other format
        out_path = osp.join(args.out, (osp.splitext(osp.basename(img_name))[0] + '.png'))
        # if out_path exists, skip

        if osp.exists(out_path):
          print(f'{out_path} exists, skipping')
        else:
          with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = inference_segmentor(model, img_path)

          # show the results
          if hasattr(model, 'module'):
              with warnings.catch_warnings():
                  warnings.filterwarnings("ignore", category=UserWarning)
                  img = model.module.show_result(img_path, result, palette=get_palette(args.palette), show=False, opacity=args.opacity)
          else:
              with warnings.catch_warnings():
                  warnings.filterwarnings("ignore", category=UserWarning)
                  img = model.show_result(img_path, result, palette=get_palette(args.palette), show=False, opacity=args.opacity)
          
          cv2.imwrite(out_path, img)

if __name__ == '__main__':
    main()

# python3 segmentation/cityscapes_seg_on_folder.py segmentation/configs/cityscapes/mask2former_beit_adapter_large_896_80k_cityscapes_ms.py segmentation/checkpoints/mask2former_beit_adapter_large_896_80k_cityscapes.pth.tar data/NeRF-OSR/Data/stjacob/final/train/rgb --out data/NeRF-OSR/Data/stjacob/final/train/cityscapes_mask

import argparse
import os
import platform
from pathlib import Path

import torch

import sys
sys.path.append('/home/zhanfeng/grasp_leaning_ws/src/yolov5/')
import detect

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
print(f'Current file path is {FILE}')

ROOT = '/home/zhanfeng/grasp_leaning_ws/src/yolov5' # YOLOv5 root directory
print(f'YOLOv5 root directory is {ROOT}')


# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


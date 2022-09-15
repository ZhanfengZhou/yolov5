import os
from pathlib import Path
import sys
sys.path.append('/home/zhanfeng/grasp_leaning_ws/src/yolov5/')
import detect_new

FILE = Path(__file__).resolve()
print(f'Current file path is {FILE}')

ROOT = '/home/zhanfeng/grasp_leaning_ws/src/yolov5' # YOLOv5 root directory
print(f'YOLOv5 root directory is {ROOT}')

ROOT_RELATIVE = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(f'YOLOv5 root relative directory is {ROOT_RELATIVE}')

#SOURCE = f'{ROOT}/test/images/img2_png_jpg.rf.bf95eb66cd5c73ccb56d68eb06227648.jpg'# file/dir/URL/glob, 0 for webcam
SOURCE = 0
print(f'Source path is {SOURCE}')


detect_new.run(
    weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
    source=SOURCE,  # file/dir/URL/glob, 0 for webcam
    data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
    imgsz=(416, 416),  # inference size (height, width)
    conf_thres=0.80,  # confidence threshold
    max_det=1,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_img=False,  # do not save images/videos
    project=f'{ROOT}/runs/detect/object_only_detect',  # save results to project/name
    name='det',  # save results to project/name
    )
import rclpy
from rclpy.node import Node
import cv2
import pyrealsense2 as rs
import os
import sys
from pathlib import Path
import torch
import PySimpleGUI as sg
from threading import Thread
import getch

from std_msgs.msg import Float32MultiArray

import os
from pathlib import Path
import sys
sys.path.append('/home/zhanfeng/grasp_learning_ws/src/yolov5/')

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadRealSense
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = '/home/zhanfeng/grasp_learning_ws/src/yolov5' # YOLOv5 root directory
ROOT_RELATIVE = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

SOURCE = 0
print(f'Source path is {SOURCE}')

@smart_inference_mode()    

class YOLOPublisher(Node):
    def __init__(self):
        super().__init__('yolo_publisher_xyz')

        self.grasp_dir = 1   # default direction is forward

        self.publisher = self.create_publisher(Float32MultiArray, 'yolo_xyz', 10)


    def run(self):
            self.run_detect(
                weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
                source=SOURCE,  # file/dir/URL/glob, 0 for webcam
                data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
                imgsz=(416, 416),  # inference size (height, width)
                conf_thres=0.80,  # confidence threshold
                max_det=1,  # maximum detections per image
                classes = [0,2,4,8],
                device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_conf=False,  # save confidences in --save-txt labels
                save_img=False,  # do not save images/videos
                project=f'{ROOT}/runs/detect/object_only_detect',  # save results to project/name
                name='det',  # save results to project/name
                )

    def set_grasp_dir(self):
        while True:
            self.grasp_dir = input('Please input grasp direction (1-forward, 2-top, 3-bottom): \n')
            
            if self.grasp_dir == '9':
                break
        return


    def grasp_center_pub(self, label, x, y, z, grasp_dir):
        print(x, y, z)
        msg = Float32MultiArray()
        msg.data = [float(x), float(y), float(z), float(grasp_dir)]  # in meter
        self.publisher.publish(msg)
        print(f'Publishing msg.data: {msg.data}')


    @smart_inference_mode() #??
    def run_detect(self,
            weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
            source=SOURCE,  # file/dir/URL/glob, 0 for webcam
            data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
            imgsz=(416, 416),  # inference size (height, width)
            conf_thres=0.75,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_img = False,  #save images/videos
            #nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            project=f'{ROOT}/runs/detect/object_only_detect',  # save results to project/name
            name='det',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)
        #save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            #dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            dataset = LoadRealSense(width=960, height=540, fps=30)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        ## Interface setup:
        if view_img:
            sg.theme("DarkBlue14")
            image_viewer_column1 = [
                [sg.Text("Real-time RGB Color Image:", size=(70, 1), font=('Helvetica', 15), justification="center")],
                [sg.Image(filename="",key="rgb")],
            ]
            image_viewer_column2 = [
                [sg.Text("Real-time Depth Image:", size=(70, 1), font=('Helvetica', 15), justification="center")],
                [sg.Image(filename="",key="depth")],
            ]
            button_column = [
                [sg.Button("Start grasp", size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Exit", size=(15, 2), font=('Helvetica', 15), pad=(0, 100))],
            ]
        
            layout = [[sg.Column(image_viewer_column1, element_justification='c'), sg.VSeperator(), sg.Column(image_viewer_column2, element_justification='c'), sg.VSeperator(),sg.Column(button_column, element_justification='c', expand_x=True)]]
            interface = sg.Window(title="Human-robot Interactive Grasp Interface", layout=layout)

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        #for data_i, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        for data_i, (path, profile, align, frames, color_depth, im, im0s, vid_cap, s) in enumerate(dataset):
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            if view_img:
                event, values = interface.read(timeout=2)

                if event == "Exit" or event == sg.WIN_CLOSED:
                    break

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                annotator_depth = Annotator(color_depth, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        c = int(cls)
                        conf = round(float(conf), 2)
                        if save_txt:  # Write to file
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            annotator_depth.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                depth_im0 = annotator_depth.result()
                    
                if view_img:
                    rgb = cv2.imencode(".png", im0)[1].tobytes()
                    depth = cv2.imencode(".png", depth_im0)[1].tobytes()
                    interface["rgb"].update(data=rgb) 
                    interface["depth"].update(data=depth) 

                    # Press enter, space or 's' to save and write the image and pub grasp center
                    if event == "Start grasp":
                        label = names[c]
                        print(f'Start grasping: Object: {label}, bounding box xywh: {xywh}, grasp direction: {self.grasp_dir}')

                        # bbox center in pixels
                        x = int(xywh[0] * 960)
                        y = int(xywh[1] * 540)

                        aligned_frames = align.process(frames)    # Align the depth frame to color frame
                        aligned_depth_frame = aligned_frames.get_depth_frame() 

                        # get the real z distance of (x, y) point
                        dis = aligned_depth_frame.get_distance(x, y)  

                        # get the real (x, y, z) value in the camera coordinates, which is a 3D array: camera_coordinate
                        # camera_coordinate[2] is still the 'dis'，camera_coordinate[0] and camera_coordinate[1] are the real x, y value
                        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics 
                        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)  

                        self.grasp_center_pub(label, camera_coordinate[0], camera_coordinate[1], dis, self.grasp_dir)


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
            
            # Print time (inference-only)
            #self.get_logger().info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        interface.close()

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        self.get_logger().info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            self.get_logger().info(f"Results saved to {colorstr('bold', save_dir)}{s}")




def main(args=None):
    rclpy.init(args=args)

    yolo_publisher_xyz = YOLOPublisher()

    thread = Thread(target=yolo_publisher_xyz.set_grasp_dir)
    thread.start()

    yolo_publisher_xyz.run()

    yolo_publisher_xyz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

from re import T
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import cv2
import pyrealsense2 as rs
import os
import sys
from pathlib import Path
import torch
import PySimpleGUI as sg
from threading import Thread

from ur_msgs.srv import YOLOOutput, Task
from example_interfaces.srv import SetBool

import os
from pathlib import Path
import sys
sys.path.append('/home/zhanfeng/grasp_learning_ws/src/yolov5/')

from models.common import DetectMultiBackend
from utils.dataloaders import LoadRealSense
from utils.general import (LOGGER, Profile,check_img_size, check_imshow, cv2,
                           increment_path, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = '/home/zhanfeng/grasp_learning_ws/src/yolov5' # YOLOv5 root directory
ROOT_RELATIVE = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@smart_inference_mode()    

class YOLOClient(Node):
    def __init__(self):
        super().__init__('yolo_client_xyz')

        self.grasp_dir = 1   # default direction is forward

        self.client = self.create_client(YOLOOutput, 'yolo_xyz', callback_group=MutuallyExclusiveCallbackGroup())
        self.client_req = YOLOOutput.Request()

        self.client_task = self.create_client(Task, 'task', callback_group=MutuallyExclusiveCallbackGroup())
        self.client_task_req = Task.Request()


    def interface_run(self):

            ## Interface setup:
            self.get_logger().info(f'Interface start:')
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
                [sg.Text("Grasp objects from table", size=(15, 2), font=('Helvetica', 15), justification="center")],
                [sg.Button("Load", key='load_table', size=(15, 2), font=('Helvetica', 15))],
                [sg.ProgressBar(5, orientation='h', size=(15, 1), border_width=4, key='load_table_bar', bar_color=("Blue","Yellow"))],
                [sg.Button("Start", key='start_table', size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Grasp", key='grasp_table', size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Stop", key='stop_table', size=(15, 2), font=('Helvetica', 15))],                
                [sg.Text("Grasp objects from human hands", size=(15, 2), font=('Helvetica', 15), justification="center", pad=(0, (25, 0)))],
                [sg.Button("Load", key='load_human', size=(15, 2), font=('Helvetica', 15))],
                [sg.ProgressBar(5, orientation='h', size=(15, 1), border_width=4, key='load_human_bar', bar_color=("Blue","Yellow"))],
                [sg.Button("Start", key='start_human', size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Grasp", key='grasp_human', size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Stop", key='stop_human', size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Flag", key='grasp_flag', size=(15, 2), font=('Helvetica', 15))],
                [sg.Button("Sleep", key='sleep', size=(15, 2), font=('Helvetica', 15), pad=(0, (25, 0)))],
                [sg.Button("Exit", key='exit',  size=(15, 2), font=('Helvetica', 15))],
            ]
        
            layout = [[sg.Column(image_viewer_column1, element_justification='c'), sg.VSeperator(), sg.Column(image_viewer_column2, element_justification='c'), sg.VSeperator(),sg.Column(button_column, element_justification='c', expand_x=True)]]
            self.interface = sg.Window(title="Human-robot Interactive Grasp Interface", layout=layout)

            while True:
                event, values = self.interface.read(timeout=2)
                if event == "exit" or event == sg.WIN_CLOSED:
                    break
                elif event == "load_human":
                    task_num = 2
                    self.send_request_task(task_num)

                    self.interface['load_human_bar'].update(1)
                    self.run_detect_from_hand(
                        weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
                        data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
                        imgsz=(416, 416),  # inference size (height, width)
                        conf_thres=0.80,  # confidence threshold
                        max_det=1,  # maximum detections per image
                        classes = [0,2,4,8],
                        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                        project=f'{ROOT}/runs/detect/object_only_detect',  # save results to project/name
                        name='det',  # save results to project/name
                        )

                elif event == "load_table":
                    task_num = 6
                    self.send_request_task(task_num)

                    self.interface['load_table_bar'].update(1)
                    self.run_detect_from_table(
                        weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
                        data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
                        imgsz=(416, 416),  # inference size (height, width)
                        conf_thres=0.40,  # confidence threshold
                        max_det=3,  # maximum detections per image
                        classes = [0,2,4,8],
                        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                        project=f'{ROOT}/runs/detect/object_on_table',  # save results to project/name
                        name='det',  # save results to project/name
                        )
                elif event == "sleep":
                    task_num = 1
                    self.send_request_task(task_num)
                else:
                    continue

            self.interface.close()

    def set_grasp_dir(self):
        while True:
            self.grasp_dir = input('Please input grasp direction (1-forward, 2-top, 3-flag grasping): \n')
            if (self.grasp_dir == '1') or (self.grasp_dir == '2') or (self.grasp_dir == '3'):
                self.get_logger().info(f'Correct input: {self.grasp_dir}')
            elif self.grasp_dir == '9':
                self.get_logger().info(f'stop grasp direction input')
                break
            else:
                self.get_logger().info(f'wrong input')
        return

    def send_request_task(self, task_num):
        # task: 1-sleep, 2-prepare grasp from human, 3-start grasping from human, 
        #   4-prepare grasp from table(including scanning) , 5-start grasping from table
        if not self.client_task.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('task service not available, wait and send request again...')
        else: 
            self.client_task_req.task = task_num
            self.get_logger().info(f'task number is {task_num}')

            self.future = self.client_task.call_async(self.client_task_req)
            # rclpy.spin_until_future_complete(self, self.future)
            #self.get_logger().info(f'task service response status: {self.future.result()}')
            rclpy.spin_once(self)
            self.get_logger().info(f'Robotic arm response to task set client')



    def send_request_yolo_output(self, x, y, z, grasp_dir):
        if not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('yolo output service not available, wait and send request again...')
        else:
            self.client_req.object_center_x = float(x)   #should be float
            self.client_req.object_center_y = float(y)
            self.client_req.object_center_z = float(z)
            self.client_req.grasp_dir = float(grasp_dir)

            self.get_logger().info(f'grasp direction: {self.client_req.grasp_dir}')

            self.future = self.client.call_async(self.client_req)
            #rclpy.spin_until_future_complete(self, self.future)
            #self.get_logger().info(f'Robotic arm response status: {self.future.result()}')
            rclpy.spin_once(self)
            self.get_logger().info(f'Robotic arm response to yolo client')


    @smart_inference_mode() 
    def run_detect_from_hand(self,
            weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
            data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
            imgsz=(416, 416),  # inference size (height, width)
            conf_thres=0.75,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            project=f'{ROOT}/runs/detect/object_only_detect',  # save results to project/name
            name='det',  # save results to project/name
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
    ):

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
        self.interface['load_human_bar'].update(2)

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        self.interface['load_human_bar'].update(3)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.interface['load_human_bar'].update(4)

        # Dataloader
        cv_check = check_imshow()
        if cv_check:
            dataset = LoadRealSense(width=960, height=540, fps=30)
        else:
            raise Exception('no camera input or cv check wrong!')
        bs = len(dataset)  # batch_size
        self.interface['load_human_bar'].update(5)
        
        # Run inference
        update_img = False
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for data_i, (path, color_depth, im, im0s, s) in enumerate(dataset):

            event, values = self.interface.read(timeout=2)
            # task_num: 0-nothing input, 1-sleep, 2-prepare grasp from human, 3-start grasping from human, 
            #   4-prepare grasp from table(including scanning) , 5-start grasping from table

            if event == "stop_human":
                task_num = 5
                self.send_request_task(task_num)
                break
            elif event == "sleep":
                task_num = 1
                self.send_request_task(task_num)
                update_img = False
            elif event == "start_human":
                task_num = 3
                self.send_request_task(task_num)
                update_img = True
            elif event == "grasp_human":
                task_num = 4
                self.send_request_task(task_num)
                update_img = True
            elif event == "grasp_flag":
                task_num = 12
                self.send_request_task(task_num)
                update_img = True
            else:
                task_num = 0   #no events, no clicks

            if not update_img:
                continue

            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)


            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
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
                        c = int(cls)
                        conf = round(float(conf), 2)

                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator_depth.box_label(xyxy, label, color=colors(c, True))


                # Stream results
                im0 = annotator.result()
                depth_im0 = annotator_depth.result()

                rgb = cv2.imencode(".png", im0)[1].tobytes()
                depth = cv2.imencode(".png", depth_im0)[1].tobytes()
                self.interface["rgb"].update(data=rgb) 
                self.interface["depth"].update(data=depth) 

                if task_num == 4:
                    self.send_request_yolo_output(0.0, 0.0, 0.0, self.grasp_dir)


        return True

    @smart_inference_mode() 
    def run_detect_from_table(self,
            weights=f'{ROOT}/runs/train/best_model_for_object_only/exp3/weights/best.pt',  # model.pt path(s)
            data=f'{ROOT}/data/coco128.yaml',  # dataset.yaml path
            imgsz=(416, 416),  # inference size (height, width)
            conf_thres=0.4,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=3,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            project=f'{ROOT}/runs/detect/object_only_detect',  # save results to project/name
            name='det',  # save results to project/name
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
    ):

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
        self.interface['load_table_bar'].update(2)

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        self.interface['load_table_bar'].update(3)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.interface['load_table_bar'].update(4)

        # Dataloader
        cv_check = check_imshow()
        if cv_check:
            dataset = LoadRealSense(width=960, height=540, fps=30)
        else:
            raise Exception('no camera input or cv check wrong!')
        bs = len(dataset)  # batch_size
        self.interface['load_table_bar'].update(5)
        
        # Run inference
        update_img = False
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for data_i, (path, color_depth, im, im0s, s) in enumerate(dataset):

            event, values = self.interface.read(timeout=2)
            # task_num: 0-nothing input, 1-sleep, 2-prepare grasp from human, 3-start grasping from human, 
            #   4-prepare grasp from table(including scanning) , 5-start grasping from table

            if event == "stop_table":
                task_num = 9
                self.send_request_task(task_num)
                break
            elif event == "sleep":
                task_num = 1
                self.send_request_task(task_num)
                update_img = False
            elif event == "start_table":
                task_num = 7
                self.send_request_task(task_num)
                self.get_logger().info(f'7 button request finished')
                update_img = True
            elif event == "grasp_table":
                task_num = 8
                self.get_logger().info(f'button grasp table is pressed, sending request')
                self.send_request_task(task_num)
                self.get_logger().info(f'8 button request finished')
                update_img = True

            else:
                task_num = 0   #no events, no clicks

            if not update_img:
                continue

            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
    
            # Inference
            with dt[1]:
                pred = model(im, augment=False, visualize=False)
    
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
    

            # Process predictions, there are multiple objects on table.
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
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
                        c = int(cls)
                        conf = round(float(conf), 2)

                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator_depth.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            depth_im0 = annotator_depth.result()

            rgb = cv2.imencode(".png", im0)[1].tobytes()
            depth = cv2.imencode(".png", depth_im0)[1].tobytes()
            self.interface["rgb"].update(data=rgb) 
            self.interface["depth"].update(data=depth) 


        return True




def main(args=None):
    rclpy.init(args=args)

    yolo_client_xyz = YOLOClient()

    thread = Thread(target=yolo_client_xyz.set_grasp_dir)
    thread.start()

    yolo_client_xyz.interface_run()
    
    yolo_client_xyz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

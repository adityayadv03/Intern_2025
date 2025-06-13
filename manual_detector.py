# To run the script Just use this file in the place of detect.py of yolov7 directory , add the centroid file in the same directory then 
# Use this command python manual_detector.py --weights yolov7.pt --img 640 --conf 0.25 --source inference/Video/People.mp4 --classes 0 --view-img
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from centroid_tracker import CentroidTracker

import psutil
import pynvml
import time

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


"""ROI_TOP_LEFT = (750, 250)
ROI_SIZE = 500  # square box
ROI_COLOR = (0, 255, 255)
inside_ids = set()"""

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    tracker = CentroidTracker(maxDisappeared=30, maxDistance=75)

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    frame_id = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        t1 = time_synchronized()
        frame_id += 1 
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                rects = []
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    rects.append((x1, y1, x2, y2))

                objects = tracker.update(rects)

                # Draw current and recently disappeared objects
                for objectID, centroid in objects.items():
                    """cx, cy = centroid
                    label = f"ID {objectID}"

                    # Draw centroid
                    cv2.circle(im0, (cx, cy), 4, (255, 0, 0), -1)

                    # Determine box size
                    box_width = 40
                    box_height = 80
                    x1, y1 = cx - box_width // 2, cy - box_height // 2
                    x2, y2 = cx + box_width // 2, cy + box_height // 2

                    # Draw bounding box and label
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Check if centroid is inside ROI
                    rx, ry = ROI_TOP_LEFT
                    if rx <= cx <= rx + ROI_SIZE and ry <= cy <= ry + ROI_SIZE:
                        inside_ids.add(objectID)
                    else:
                        inside_ids.discard(objectID)"""
                    
                    cx, cy = centroid
                    label = f"ID {objectID}"

                    # Draw centroid
                    cv2.circle(im0, (cx, cy), 4, (255, 0, 0), -1)

                    # Check if the object is disappeared
                    disappeared_frames = tracker.disappeared.get(objectID, 0)
                    color = (128, 128, 128) if disappeared_frames > 0 else (255, 0, 0)

                    # More realistic human box size (width 40, height 80)
                    box_width = 40
                    box_height = 80
                    x1, y1 = cx - box_width // 2, cy - box_height // 2
                    x2, y2 = cx + box_width // 2, cy + box_height // 2

                    # Draw bounding box and label
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                
                """for objectID, centroid in objects.items():
                    cx, cy = centroid
                    label = f"ID {objectID}"
                    cv2.circle(im0, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.putText(im0, label, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"""
            
            """# Draw the ROI square
            rx, ry = ROI_TOP_LEFT
            cv2.rectangle(im0, (rx, ry), (rx + ROI_SIZE, ry + ROI_SIZE), ROI_COLOR, 2)

            # Show count of objects inside ROI
            count_label = f"Count: {len(inside_ids)}"
            cv2.putText(im0, count_label, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)"""
            
            # ----- System Performance Overlay -----
            inference_time = t2 - t1
            fps = 1.0 / inference_time if inference_time > 0 else 0.0

            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            # GPU metrics using pynvml
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_percent = int((gpu_mem.used / gpu_mem.total) * 100)

            # Benchmark info text
            benchmark_texts = [
                f"Inference FPS: {fps:.2f}",
                f"Frame #: {frame_id}",
                f"CPU Load: {cpu_usage:.1f}%",
                f"RAM Usage: {ram_usage:.1f}%",
                f"GPU Load: {gpu_util}%",
                f"GPU Mem: {gpu_mem_percent}%",
            ]

            # Draw the benchmark texts on video frame (top-left corner)
            start_x, start_y = 10, 30
            line_height = 25
            for i, text in enumerate(benchmark_texts):
                y = start_y + i * line_height
                cv2.putText(im0, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

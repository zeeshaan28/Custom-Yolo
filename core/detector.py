import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .models.experimental import attempt_load
from .utils.general import non_max_suppression, scale_coords, xyxy2xywh, increment_path, colors, plot_one_box, letterbox
from .utils.torch_utils import select_device


model = None
initialized = None
iou_thres= 0.4
device = None
precision = None
line_thickness=3
frame_counter=0
save_dir=None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def initialize():
    
    global model, initialized, iou_thres, device, precision, save_dir
    
    # device = os.getenv('DEVICE', 'cpu')
    device = select_device()
    precision = os.getenv('PRECISION', 'float32')

    print('Running initialization...')
    print(f'Device    : {device}')
    print(f'Precision : {precision}')
    
    print('\nSkyLark Labs Detection System\n')
           
    yolo_weights = os.path.join(BASE_DIR, 'parameters', 'skylark_yolo.pt')
    model = attempt_load(yolo_weights, map_location=device)
    model.eval()

    if precision == 'float16':
        assert device == 'cuda', f'\n\nHalf precision is only available when device is set to "cuda", but got "{device}".\n'
        model = model.half()
     
    # Directories
    out='outputs'
    save_dir = increment_path(Path(out) / 'inf', exist_ok=False)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)
    
    initialized = True


@torch.no_grad()
def detect(source=None,
           read_frame=10,
           conf_thres=0.25,
           save_txt=True,
           show_results=True):

    global model, initialized, iou_thres, device, precision, line_thickness, frame_counter ,save_dir
    
    # Initialising
    if not initialized:
        initialize()
        
    
    names = model.names

    # Dataloader
    cam_ids = source['camera_id']

    path = cam_ids
    
    num_sources=len(path)
    im0s = source['frames']  

    img = [letterbox(x, 640, stride=32)[0] for x in im0s]
    img = np.stack(img, 0)
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run inference
    pred = model(img)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        
        p, s, im0 = path[i], f'{i}: ', im0s[i].copy()
            
        p = Path(p)  # to Path
     
        txt_path = os.path.join(str(Path(save_dir)), p) + '_' + str(frame_counter)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):

                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = names[int(cls)] +'\t'
                    line += ' '
                        # print(xywh, type(xywh))
                    line += ' '.join(str(e) for e in xywh)
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(line + '\n')
                        
                    cv2.imwrite(txt_path + '.png', im0s[i])

                if show_results:
                    c = int(cls)  # integer class
                    label = (f'{names[c]} {conf:.2f}')
                    im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)

            # Stream results
            if show_results:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


    frame_counter += 1        

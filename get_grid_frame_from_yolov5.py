import sys
sys.path.append('/Users/yuqiao/Desktop/project2/yolov5')  # 修改为 yolov5 目录的绝对路径

import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.dataloaders import LoadImages
from pathlib import Path
from PIL import Image

def get_grid_frame_from_yolov5(source):
    # Initialize
    weights = 'yolov5/runs/train/exp3/weights/best.pt'  
    img_size = 640  # size of the image
    original_img = Image.open(source) # for cropping
    img = Image.open(source)
    width, height = img.size

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt

    # Load the data
    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=pt)

    # To Predict
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # Normalize the image to the range [0,1].
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Predict
        pred = model(img)

        # Non-Maximum Suppression (NMS)  
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # detections per image
        for i, det in enumerate(pred):  
            p, im0 = Path(path), im0s.copy()

            if len(det):
                # Scale the coordinates from the inferred image size back to the original image size.
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Find the target with the highest confidence.
                det_max_conf = det[det[:, 4].argmax()] 

                # Extract the bounding box coordinates, confidence score, and category.
                xyxy = det_max_conf[:4]
                conf = det_max_conf[4]
                cls = det_max_conf[5]
                
                # Extract the specific coordinates of the bounding box with the highest confidence.
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # print(f"Bounding box coordinates: Top-left ({x1}, {y1}), Bottom-right ({x2}, {y2})")

                # Ensure the result is inside the image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                # print(x1, y1, x2, y2)

                cropped_img = original_img.crop((x1, y1, x2, y2))

                # Save the cropped image for debugging purposes.
                cropped_img.save(f"output/frame.png")

        return x1, y1, x2, y2
import os
import json
import subprocess
import cv2

def get_box_corners(box, image_width, image_height):
    center_x, center_y, width, height = box
    center_x = max(0, min(center_x, 1))
    center_y = max(0, min(center_y, 1))
    width = max(0, min(width, 1))
    height = max(0, min(height, 1))
    
    x1 = int((center_x - width / 2) * image_width)
    y1 = int((center_y - height / 2) * image_height)
    x2 = int((center_x + width / 2) * image_width)
    y2 = int((center_y + height / 2) * image_height)
    
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def run_yolo_detection(image_path, config_path, weights_path, data_path, json_output_path, thresh=0.3):
    command = [
        './darknet/darknet', 'detector', 'test', data_path, config_path, weights_path, image_path, 
        '-dont_show', '-ext_output', '-out', json_output_path, '-thresh', str(thresh)
    ]
    subprocess.run(command, stdout=subprocess.PIPE)

def parse_yolo_output_from_json(json_path, image_width, image_height):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    boxes = []
    for detection in data[0]['objects']:
        box = detection['relative_coordinates']
        center_x = box['center_x']
        center_y = box['center_y']
        width = box['width']
        height = box['height']
        boxes.append((center_x, center_y, width, height))
    return boxes

def crop_and_save(image_path, corners, output_path):
    image = cv2.imread(image_path)
    x1, y1 = corners[0]
    x2, y2 = corners[2]
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

    return (x1, y1, x2, y2), cropped_image

def crop_image_from_yolo(image_path, output_dir,
                         config_path='config/yolov3_custom.cfg', 
                         weights_path='models/yolov3_custom_final.weights', 
                         data_path='config/obj.data', 
                         thresh=0.3):
    
    # 生成唯一的 JSON 文件路径
    json_path = os.path.join(output_dir, f'{os.path.basename(image_path).split(".")[0]}_result.json')
   
    # Load the image to get its dimensions
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Run YOLOv3 detection and save results to JSON
    run_yolo_detection(image_path, config_path, weights_path, data_path, json_path, thresh)
    
    # Parse the YOLOv3 output from JSON file
    boxes = parse_yolo_output_from_json(json_path, image_width, image_height)

    croped_points = []
    croped_images = []
    
    # Calculate corners for each bounding box and save cropped images
    for i, box in enumerate(boxes):
        corners = get_box_corners(box, image_width, image_height)
        output_path = os.path.join(output_dir, f'cropped_{os.path.basename(image_path).split(".")[0]}_{i}.png')
        croped_point, croped_image = crop_and_save(image_path, corners, output_path)
        
        croped_points.append(croped_point)
        croped_images.append(croped_image)

    return croped_points, croped_images

def batch_process_images(input_folder, output_folder, config_path, weights_path, data_path, thresh=0.3):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path}...")
            
            # Process the image and save the cropped results
            crop_image_from_yolo(image_path, output_folder, config_path, weights_path, data_path, thresh)

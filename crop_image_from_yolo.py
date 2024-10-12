import json
import subprocess
import cv2

def get_box_corners(box, image_width, image_height):
    """
    Given a bounding box in YOLO format (center_x, center_y, width, height),
    compute the coordinates of the four corners of the bounding box.
    """
    center_x, center_y, width, height = box

    
    center_x = max(0, min(center_x, 1))
    center_y = max(0, min(center_y, 1))
    width = max(0, min(width, 1))
    height = max(0, min(height, 1))

    x1 = int((center_x - width / 2) * image_width)
    y1 = int((center_y - height / 2) * image_height)
    x2 = int((center_x + width / 2) * image_width)
    y2 = int((center_y + height / 2) * image_height)

    # Constrain the coordinates to remain within the image boundaries.
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def run_yolo_detection(image_path, config_path, weights_path, data_path, thresh=0.3):
    """
    Run YOLOv3 detection on the given image and save the result to a JSON file.
    """
    command = [
        './darknet/darknet', 'detector', 'test', data_path, config_path, weights_path, image_path, 
        '-dont_show', '-ext_output', '-out', 'output/result.json', '-thresh', str(thresh)
    ]
    subprocess.run(command, stdout=subprocess.PIPE)

def parse_yolo_output_from_json(json_path, image_width, image_height):
    """
    Parse YOLOv3 output from the JSON file to extract bounding boxes.
    """
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
    """
    Crop the image using the given corners and save the result.
    """
    image = cv2.imread(image_path)
    x1, y1 = corners[0]
    x2, y2 = corners[2]
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

    return (x1, y1, x2, y2), cropped_image

def crop_image_from_yolo(image_path, 
                         config_path='config/yolov3_custom.cfg', 
                         weights_path='models/yolov3_custom_final.weights', 
                         data_path='config/obj.data', 
                         json_path='output/result.json', 
                         thresh=0.3):

    # Load the image to get its dimensions
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Run YOLOv3 detection and save results to JSON
    run_yolo_detection(image_path, config_path, weights_path, data_path, thresh)
    
    # Print the JSON file content for debugging
    with open(json_path, 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))
    
    # Parse the YOLOv3 output from JSON file
    boxes = parse_yolo_output_from_json(json_path, image_width, image_height)
    print("Parsed Boxes:\n", boxes)

    croped_points = []
    croped_images = []
        
    # Calculate corners for each bounding box and save cropped images
    for i, box in enumerate(boxes):
        corners = get_box_corners(box, image_width, image_height)
        print(f"Bounding box corners: {corners}")
        output_path = f'output/cropped_{i}.png'
        croped_point, croped_image = crop_and_save(image_path, corners, output_path)
        print(f"Cropped image saved to: {output_path}")

        croped_points.append(croped_point)
        croped_images.append(croped_image)

    return croped_points, croped_images

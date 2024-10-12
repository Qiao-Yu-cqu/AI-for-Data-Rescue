import json
import subprocess
import cv2

def get_box_corners(box, image_width, image_height):
    """
    给定YOLO格式的边界框（中心坐标，宽度，高度），计算边界框的四个角的坐标。
    """
    center_x, center_y, width, height = box

    # 保证中心点和宽高比例在合理范围内
    center_x = max(0, min(center_x, 1))
    center_y = max(0, min(center_y, 1))
    width = max(0, min(width, 1))
    height = max(0, min(height, 1))

    # 计算边界框的左上角和右下角的绝对坐标
    x1 = int((center_x - width / 2) * image_width)
    y1 = int((center_y - height / 2) * image_height)
    x2 = int((center_x + width / 2) * image_width)
    y2 = int((center_y + height / 2) * image_height)

    # 限制坐标在图片范围内
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def run_yolo_detection(image_path, config_path, weights_path, data_path, thresh=0.3):
    """
    运行YOLOv3检测并将结果保存到JSON文件。
    """
    command = [
        './darknet/darknet', 'detector', 'test', data_path, config_path, weights_path, image_path, 
        '-dont_show', '-ext_output', '-out', 'output/result.json', '-thresh', str(thresh)
    ]
    subprocess.run(command, stdout=subprocess.PIPE)

def parse_yolo_output_from_json(json_path, image_width, image_height):
    """
    从JSON文件中解析YOLOv3输出并提取边界框。
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
    使用给定的角点裁剪图像并保存裁剪结果。
    """
    image = cv2.imread(image_path)
    x1, y1 = corners[0]
    x2, y2 = corners[2]
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

    return (x1, y1, x2, y2), cropped_image

def crop_image_from_yolo(image_path, 
                         config_path='config/yolov3_custom_frame.cfg', 
                         weights_path='models/yolov3_custom_last.weights', 
                         data_path='config/obj_frame.data', 
                         json_path='output/result_frame.json', 
                         thresh=0.3):

    # 加载图像获取其尺寸
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # 运行YOLOv3检测并将结果保存为JSON文件
    run_yolo_detection(image_path, config_path, weights_path, data_path, thresh)
    
    # 打印JSON文件内容以便调试
    with open(json_path, 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))
    
    # 解析YOLOv3输出的边界框
    boxes = parse_yolo_output_from_json(json_path, image_width, image_height)
    print("Parsed Boxes:\n", boxes)

    cropped_points = []
    cropped_images = []
        
    # 计算每个边界框的角点并保存裁剪后的图像
    for i, box in enumerate(boxes):
        corners = get_box_corners(box, image_width, image_height)
        print(f"Bounding box corners: {corners}")
        output_path = f'cropped_{i}.png'
        cropped_point, cropped_image = crop_and_save(image_path, corners, output_path)
        print(f"Cropped image saved to: {output_path}")

        cropped_points.append(cropped_point)
        cropped_images.append(cropped_image)

    return cropped_points, cropped_images

# # 使用示例：
# points, images = crop_image_from_yolo(image_path='/Users/yuqiao/Desktop/project2/example/4010_1966_051_page_1.jpg')
# print(points)

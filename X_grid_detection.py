import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 使用和之前代码相同的方法检测竖直线
def detect_vertical_lines(image_path):
    src = cv2.imread(image_path)
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bin_src = cv2.adaptiveThreshold(~gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    hline = cv2.getStructuringElement(cv2.MORPH_RECT, (src.shape[1] // 16, 1))
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, src.shape[0] // 16))
    tmp = cv2.erode(bin_src, vline)
    vert_lines = cv2.dilate(tmp, vline)
    vert_lines = cv2.blur(vert_lines, (3, 3))
    tmp = cv2.erode(bin_src, hline)
    hor_lines = cv2.dilate(tmp, hline)
    hor_lines = cv2.blur(hor_lines, (3, 3))
    lines = cv2.addWeighted(vert_lines, 1, hor_lines, 1, 0)
    edges = cv2.Canny(lines, 50, 150, apertureSize=3)

    lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    detected_lines = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 5:  # 近似竖直线
                detected_lines.append((x1 + x2) // 2)

    return detected_lines

# 绘制图像并根据检测到的竖线进行微调
def draw_adjusted_lines(image_path, start_x, interval, tolerance=5):
    image = Image.open(image_path)
    width, height = image.size
    detected_lines = detect_vertical_lines(image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # 计算所有初始竖线的位置
    original_x_positions = list(range(start_x, width, interval))
    
    adjusted_x_positions = []
    for x in original_x_positions:
        adjusted_x = x
        for detected_x in detected_lines:
            if abs(x - detected_x) <= tolerance:
                adjusted_x = detected_x
                break
        adjusted_x_positions.append(adjusted_x)

    # 绘制调整后的竖线
    for x in adjusted_x_positions:
        plt.axvline(x=x, color='green', linestyle='--', linewidth=1)

    plt.show()

    return adjusted_x_positions

# 运行代码
image_path = '/Users/yuqiao/Desktop/project2/example8.jpg'  # 替换为你的图像路径
start_x = 5 # 起始x坐标
interval = 34  # 间隔
tolerance = interval / 2  # 调整公差

lst = draw_adjusted_lines(image_path, start_x, interval, tolerance)
print(len(lst))

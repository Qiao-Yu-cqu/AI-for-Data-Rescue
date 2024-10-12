import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像尺寸
width = 2048  # 更宽的图像来表示一周的变化
height = 512  # 高度不变

# 创建空白图像（黑色背景）
img = np.zeros((height, width), dtype=np.uint8)  # 黑色背景

# 设置控制点的数量 (模拟一周的水位变化)
num_days = 7
num_points_per_day = 15  # 每一天的变化细节
total_points = num_days * num_points_per_day

# x 坐标的分布
x_points = np.linspace(0, width, total_points)

# y 坐标的分布，模拟每天的水位高度变化
y_points_main = []
y_points_secondary = []
for day in range(num_days):
    # 每天水位高度在一定范围内随机波动
    base_height_main = np.random.randint(200, height - 200)  # 主线基准水位高度
    base_height_secondary = np.random.randint(200, height - 200)  # 次线基准水位高度
    
    daily_variation_main = np.random.normal(0, 5, num_points_per_day)  # 主线波动幅度
    daily_variation_secondary = np.random.normal(0, 7, num_points_per_day)  # 次线波动幅度
    
    y_day_points_main = base_height_main + daily_variation_main
    y_day_points_secondary = base_height_secondary + daily_variation_secondary
    
    y_points_main.extend(y_day_points_main)
    y_points_secondary.extend(y_day_points_secondary)

# 将 x 和 y 组合成控制点 (确保它们是 (x, y) 对)
points_main = np.array(list(zip(x_points, y_points_main)), dtype=np.int32)
points_secondary = np.array(list(zip(x_points, y_points_secondary)), dtype=np.int32)

# 确定是否出现断线
draw_broken_line = np.random.rand() < 0.02  # 2%概率出现断线

# 选择随机缺失的天数（一或两天）
if draw_broken_line:
    missing_days = np.random.choice([1, 2])  # 随机缺失一天或两天的信息
    missing_start_day = np.random.randint(0, num_days - missing_days)  # 随机选择缺失开始的天数

    # 确定缺失的时间段的范围
    missing_start_point = missing_start_day * num_points_per_day
    missing_end_point = (missing_start_day + missing_days) * num_points_per_day
else:
    missing_start_point = None
    missing_end_point = None

# 绘制主线
for i in range(len(points_main) - 1):
    # 绘制主线，如果不是在缺失区间
    if missing_start_point is None or not (missing_start_point <= i < missing_end_point):
        point1 = tuple(points_main[i])
        point2 = tuple(points_main[i + 1])
        thickness = np.random.randint(2, 10)  # 主线的粗细变化
        img = cv2.line(img, point1, point2, color=255, thickness=thickness)  # 白色线条

# 确定是否出现两条线
draw_two_lines = np.random.rand() < 0.02  # 2%概率出现双线

# 如果有双线，绘制次线
if draw_two_lines:
    for i in range(len(points_secondary) - 1):
        # 绘制次线，次线的波动与主线不同，也不一定缺失相同的天数
        if missing_start_point is None or not (missing_start_point <= i < missing_end_point):
            point1 = tuple(points_secondary[i])
            point2 = tuple(points_secondary[i + 1])
            thickness = np.random.randint(2, 10)  # 次线的粗细变化
            img = cv2.line(img, point1, point2, color=255, thickness=thickness)  # 白色线条

# 压缩图像到512x512
img_resized = cv2.resize(img, (512, 512))

# 保存图像
cv2.imwrite('water_level_simulation_white_lines_on_black.png', img_resized)

# 显示图像
plt.imshow(img_resized, cmap='gray')
plt.axis('off')
plt.show()

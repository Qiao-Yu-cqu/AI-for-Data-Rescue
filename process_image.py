import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from PIL import Image
from scripts.crop_image_from_yolo import crop_image_from_yolo


def convert_to_grayscale(image):
    # Convert the image into grayscale
    grayscale_img = image.convert("L")
    return grayscale_img

def plot_histogram(grayscale_img):
    # plot the bar chart of gray_img
    image_array = np.array(grayscale_img)
    histogram, bin_edges = np.histogram(image_array, bins=256, range=(0, 255))
    plt.figure()
    plt.bar(bin_edges[:-1], histogram, width=1, edgecolor='black')
    plt.title('Pixel Distribution in Grayscale Image')
    plt.xlabel('Pixel value (0-255)')
    plt.ylabel('Pixel count')
    plt.show()

def threshold_image(grayscale_img, low_threshold, high_threshold):
    # To filter the image according to the threshold
    img_array = np.array(grayscale_img)
    thresholded_array = np.where((img_array >= low_threshold) & (img_array <= high_threshold), 255, 0)
    thresholded_img = Image.fromarray(thresholded_array.astype(np.uint8))
    return thresholded_img

def median_filter_3x3(pil_image):
    image_array = np.array(pil_image)
    filtered_image = cv2.medianBlur(image_array, 1)  # Change kernel size to 3
    return filtered_image

def apply_closing_operation(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def apply_open_operation(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return closed_image

def find_first_sudden_increase_point(histogram, threshold):
    # Calculate the difference between two adjacent points.
    diff = np.diff(histogram)
    
    # Find the first mutation point that exceeds the threshold.
    for i, value in enumerate(diff):
        if value > threshold:
            return i
    return None

def filter_image(image_path):
    points, images = crop_image_from_yolo(image_path=image_path)
    detected_num = len(points)
    closed_pil_images = []
    for idx in range(detected_num):
        cropped_image_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_image_rgb)

        # Load the image and process it.
        grayscale_image = convert_to_grayscale(pil_image)
        plot_histogram(grayscale_image)  # 如果需要看直方图，确保这一行被调用
        image_array = np.array(grayscale_image)
        histogram, bin_edges = np.histogram(image_array, bins=256, range=(0, 255))

        # Line is always the lowest gray value
        thre = 0.00007 * np.prod(image_array.shape)

        # Find the first mutation point.
        sudden_increase_index = find_first_sudden_increase_point(histogram, thre)
        # print("sudden_increase_index = ", sudden_increase_index)

        # upper_th, lower_th = threshold[1], threshold[0]
        thresholded_image = threshold_image(grayscale_image, 0, sudden_increase_index)
        filtered_image = median_filter_3x3(thresholded_image)    # useless
        closed_image = apply_closing_operation(filtered_image)     # to connect the small points

        # 使用 PIL 显示图像
        closed_pil_image = Image.fromarray(closed_image)
        # print(closed_pil_image)
        closed_pil_images.append(closed_pil_image)
        closed_pil_image.save(f"output/result_1.png")
        # closed_pil_image.show() 
    # print("len(closed_pil_images): ", len(closed_pil_images))
    # print("points: ", points)
    return closed_pil_images, points

def evaluate_coverage(mask):
    # To caulate the coverage of the line in the whole line
    non_zero_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage_percentage = (non_zero_pixels / total_pixels) * 100
    return coverage_percentage

def filter_image2(image_path):
    points, images = crop_image_from_yolo(image_path=image_path)
    detected_num = len(points)
    closed_pil_images = []
    
    # 动态颜色范围调整参数
    # According to the image
    lower_h, upper_h = 100, 130
    lower_s, upper_s = 50, 255
    lower_v, upper_v = 50, 255
    # Define the step of each iteration
    step_h = 5
    step_s = 20
    step_v = 20
    coverage_threshold = 0.7
    max_iterations = 10

    for idx in range(detected_num):
        cropped_image_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_image_rgb)
        
        # 将图像转换为HSV
        resized_image = cv2.resize(images[idx], (int(images[idx].shape[1] / 2), int(images[idx].shape[0] / 2)))
        hsv_resized = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

        # 动态调整颜色范围提取掩码
        for iteration in range(max_iterations):
            lower_bound = np.array([lower_h, lower_s, lower_v])
            upper_bound = np.array([upper_h, upper_s, upper_v])
            mask = cv2.inRange(hsv_resized, lower_bound, upper_bound)

            # 计算覆盖率
            coverage_score = evaluate_coverage(mask)

            # 如果覆盖率达到阈值，停止迭代
            if coverage_score >= coverage_threshold:
                break

            # 否则继续扩展颜色范围
            lower_h = max(0, lower_h - step_h)
            upper_h = min(180, upper_h + step_h)
            lower_s = max(0, lower_s - step_s)
            upper_s = min(255, upper_s + step_s)
            lower_v = max(0, lower_v - step_v)
            upper_v = min(255, upper_v + step_v)

        # 将掩码扩展回原始图像大小
        mask_upscaled = cv2.resize(mask, (images[idx].shape[1], images[idx].shape[0]), interpolation=cv2.INTER_NEAREST)
        line_extracted = cv2.bitwise_and(images[idx], images[idx], mask=mask_upscaled)
        
        # 将提取的结果转换为灰度图像
        grayscale_image = cv2.cvtColor(line_extracted, cv2.COLOR_BGR2GRAY)

        # 中值滤波
        filtered_image = median_filter_3x3(grayscale_image)
        
        # 闭运算
        closed_image = apply_closing_operation(filtered_image)

        # 转换为PIL格式并保存结果
        closed_pil_image = Image.fromarray(closed_image)
        closed_pil_images.append(closed_pil_image)
        closed_pil_image.save(f"output/result_{idx + 1}.png")
    
    return closed_pil_images, points

def process_image(image_path, color_style=False):
    """
    Processes an image to extract the main curve's points, filter noise, and interpolate the curve. 
    The curve is smoothed and drawn on the original image. The processed image is then saved to a file.

    Args:
    image_path (str): The file path to the input image to be processed.

    Returns:
    tuple: A tuple containing the following:
        - all_cols (list): The x-coordinates of the columns in the image after processing.
        - smoothed_points (list): The y-coordinates of the smoothed curve points.
        - x1 (int): The x-coordinate of the top-left corner of the detected frame.
        - y1 (int): The y-coordinate of the top-left corner of the detected frame.
        - x2 (int): The x-coordinate of the bottom-right corner of the detected frame.
        - y2 (int): The y-coordinate of the bottom-right corner of the detected frame.

    Notes:
    - The function starts by filtering the image and converting it into a binary format.
    - The function scans each column of the image to detect white pixels and determine the median position of the curve.
    - Outliers are filtered, and a Univariate Spline is applied to interpolate missing points.
    - The curve is smoothed using a Gaussian filter, and the result is drawn on the original image.
    - The processed image with the drawn curve is saved as a new file.
    """
    original_image = cv2.imread(filename=image_path)

    if color_style:
        closed_pil_images, frame = filter_image2(image_path=image_path)
    else:
        closed_pil_images, frame = filter_image(image_path=image_path)

    closed_image_array = np.array(closed_pil_images[0])
    # print("frame:", frame)
    # Ensure binary image
    _, binary_image = cv2.threshold(closed_image_array, 127, 255, cv2.THRESH_BINARY)

    # Get the dimensions of the image
    height, width = binary_image.shape

    # Initialize lists to hold the main curve points
    curve_points = []

    # Threshold for noise removal (distance from median)
    distance_threshold = 10

    # Scan each column
    for col in range(width):
        # Get the white pixels in this column
        white_pixels = np.where(binary_image[:, col] == 255)[0]
        
        if len(white_pixels) > 0:
            # Calculate the median position of the white pixels
            median_pos = int(np.median(white_pixels))
            
            # Filter out points that are too far from the median
            filtered_pixels = [pixel for pixel in white_pixels if abs(pixel - median_pos) <= distance_threshold]
            
            if len(filtered_pixels) > 0:
                # Use the median of the filtered pixels as the curve point
                curve_point = int(np.median(filtered_pixels))
                curve_points.append((col, curve_point))
            else:
                curve_points.append((col, median_pos))
        else:
            # No white pixels in this column, skip
            curve_points.append((col, None))

    # Filter out None values from curve_points
    filtered_curve_points = [(col, point) for col, point in curve_points if point is not None]

    # Separate columns and points into two lists
    cols, points = zip(*filtered_curve_points)

    # Remove outliers using a moving average filter
    window_size = 10
    filtered_points = []

    for i in range(len(points)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(points), i + window_size // 2)
        window_points = points[window_start:window_end]
        median = np.median(window_points)
        if abs(points[i] - median) <= distance_threshold:
            filtered_points.append(points[i])
        else:
            filtered_points.append(median)

    # Use UnivariateSpline for spline interpolation
    # Insert the values and soomth the points
    spline = UnivariateSpline(cols, filtered_points, s=1)

    # Generate new y-values for each column using the spline
    spline_points = spline(cols)

    # Fill missing values using linear interpolation
    interp_func = interp1d(cols, spline_points, kind='linear', fill_value='extrapolate')
    all_cols = np.arange(width)
    filled_spline_points = interp_func(all_cols)

    # Smooth the curve using a Gaussian filter
    smoothed_points = gaussian_filter1d(filled_spline_points, sigma=2)

    x1, y1, x2, y2 = frame[0][0], frame[0][1], frame[0][2], frame[0][3]
    # print("height = ", height)
    smoothed_points = [point + y1 for point in smoothed_points]
    all_cols = [col+x1 for col in all_cols]
    # print(smoothed_points)

    # # Create a new blank image to draw the final curve
    # final_curve_image = np.zeros_like(binary_image)

    # Draw the smoothed curve on the new image
    for col, point in zip(all_cols, smoothed_points):
        original_image[int(point), col] = (0, 255, 0)
        original_image[int(point)-1, col] = (0, 255, 0)
        original_image[int(point)+1, col] = (0, 255, 0)
        
    closed_pil_image = Image.fromarray(original_image)
    # closed_pil_image.show()
    output_path = f'output/closed_image.png'
    closed_pil_image.save(output_path)

    # print(f"Image saved at {output_path}")

    return all_cols, smoothed_points, x1, y1, x2, y2

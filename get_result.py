from scripts.get_grid_frame_from_yolov5 import get_grid_frame_from_yolov5
from scripts.process_image import process_image
from PIL import Image, ImageDraw
from math import modf
from scripts.openaiapi import call_openai_api
from dateutil import parser
from scripts.sample_interval_time import sample_interval
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Define constants for parameter tuning
ADAPTIVE_THRESH_BLOCK_SIZE = 15
ADAPTIVE_THRESH_C = -2
VERTICAL_KERNEL_RATIO = 16
HORIZONTAL_KERNEL_RATIO = 16
BLUR_KERNEL_SIZE = (3, 3)
HOUGH_THRESHOLD = 100
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 10
VERTICAL_LINE_THRESHOLD = 5

def detect_vertical_lines(image_path):
    """
    Detects vertical lines in an image using adaptive binarization, morphological operations, and Hough transform.

    Args:
    image_path (str): The path to the image file.

    Returns:
    list: A list of x-coordinates representing the detected vertical lines' positions.
    """
    # Read the image and apply adaptive binarization, which can handle uneven lighting conditions.
    src = cv2.imread(image_path)
    image_width, image_height = src.shape[1], src.shape[0]
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bin_src = cv2.adaptiveThreshold(~gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)

    # Obtain vertical and horizontal structuring elements for subsequent morphological operations.
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, (src.shape[1] // HORIZONTAL_KERNEL_RATIO, 1))
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, src.shape[0] // VERTICAL_KERNEL_RATIO))

    # Use erosion and dilation to remove smaller non-vertical structures (noise).
    tmp = cv2.erode(bin_src, vline)     # remove the small vertical lines
    vert_lines = cv2.dilate(tmp, vline) # recover the wrong removed lines

    # Apply mean blur to smooth the extracted vertical lines.
    vert_lines = cv2.blur(vert_lines, BLUR_KERNEL_SIZE)

    # The same method to extract horizontal lines
    tmp = cv2.erode(bin_src, hline)
    hor_lines = cv2.dilate(tmp, hline)
    hor_lines = cv2.blur(hor_lines, BLUR_KERNEL_SIZE)

    # Mix vertical and horizontal lines and get the edges
    lines = cv2.addWeighted(vert_lines, 1, hor_lines, 1, 0)
    edges = cv2.Canny(lines, 50, 150, apertureSize=3) # Canny detection methods

    # Use the probabilistic Hough transform to detect lines and return their positions.
    lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD, 
                              minLineLength=int(min(image_width, image_height) / 5), maxLineGap=int(min(image_width, image_height) / 20))

    # If the difference between x1 and x2 is less than the vertical line threshold, consider it a vertical line.
    detected_lines = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < VERTICAL_LINE_THRESHOLD:  # Approximate vertical line
                detected_lines.append((x1 + x2) // 2)

    return detected_lines


def decimal_to_feet_inches(decimal_feet):
    """
    Converts a decimal representation of feet into feet and inches.

    Args:
    decimal_feet (float): The length in decimal feet.

    Returns:
    tuple: A tuple containing the integer feet and the rounded inches (up to two decimal places).
    """

    feet = int(decimal_feet)
    fractional, whole = modf(decimal_feet)
    
    inches = fractional * 12
    return feet, round(inches, 2)

def convert_date_to_numeric(date_str, dayfirst=True):
    """
    Converts a date string into a standardized numeric date format (day, abbreviated month, year).

    Args:
    date_str (str): The date string to be converted.
    dayfirst (bool): Whether to parse the date with day first (default is True).

    Returns:
    str: The formatted date string in 'DD Mon YYYY' format.
    """

    date_obj = parser.parse(date_str, dayfirst=dayfirst)
    
    formatted_date = date_obj.strftime('%d %b %Y')
    
    return formatted_date

def split_number_and_unit(input_str):
    """
    Splits a string into its numeric value and unit using regular expressions.

    Args:
    input_str (str): The input string containing a number and a unit.

    Returns:
    tuple: A tuple containing the numeric value (as a string) and the unit (as a string). 
           Returns (None, None) if the input string doesn't match the expected pattern.
    """
    # Use a regular expression to extract numbers and units.
    match = re.match(r"(\d+\.?\d*)\s*(\w+)", input_str)
    
    if match:
        number = match.group(1)
        unit = match.group(2)
        return number, unit
    else:
        return None, None


def get_result(image_path,
               interval = None,
               start_date = None,
               end_date = None,
               max_height = None,
               unit = None,
               days_recorded = None,
               adjust = True,
               color_style = False):
    """
    Analyzes a hydroelectric chart image to extract grid line information, 
    recorded time intervals, and height measurements in feet and inches. 
    It uses various APIs (OpenAI and Microsoft), image processing, and date manipulation.

    Args:
    image_path (str): The file path to the chart image.
    interval (float, optional): Time interval between measurements in hours. 
                                If None, this value is determined automatically.
    start_date (str, optional): Start date of the recording period in 'DD Mon YYYY' format. 
                                If None, the date is extracted automatically.
    end_date (str, optional): End date of the recording period in 'DD Mon YYYY' format. 
                              If None, the date is extracted automatically.
    max_height (str, optional): Maximum height value shown on the chart background. 
                                If None, the value is extracted automatically.
    unit (str, optional): The unit of measurement for height (e.g., feet, meters).
                          If None, it is extracted automatically.
    days_recorded (int, optional): Number of days recorded in the chart. 
                                   If None, the value is extracted automatically.
    adjust (bool, optional): If True, adjusts vertical grid lines based on detected lines. 
                             Default is True.

    Returns:
    None: The function processes the chart image and writes the extracted time intervals and 
          height measurements to an Excel file named 'output.xlsx'. Additionally, an image with 
          detected vertical lines is saved as 'image_with_lines.jpg'.
    """
    
    # Obtain the coordinates of the top-left and bottom-right corners of the background grid lines.
    x1_frame, y1_frame, x2_frame, y2_frame = get_grid_frame_from_yolov5(source=image_path)

    # Obtain the x and y coordinates of the detected curve in the image, with the top-left corner of the image as the origin.
    all_cols, smoothed_points, x1_line, y1_line, x2_line, y2_line = process_image(image_path=image_path, color_style=color_style)

    # Call the OpenAI API to analyze the image, and include key information such as time, location, etc. in the JSON file.
    openai_json = call_openai_api(image_path=image_path)
    
    # Call Microsoft's API to obtain the time intervals in the chart, the time representation format, and the distance between each number on the x-axis.
    interval_sample, time_format, gap_interval = sample_interval(image_path=image_path)

    # If an interval is provided, use the given interval. Otherwise, use the automatically obtained interval.
    if interval is None:
        interval = interval_sample
    else:
        interval = interval

    # Parse the JSON file returned from OpenAI to extract basic information from the image. Translate the key details.
    if start_date is None:
        start_date = str(convert_date_to_numeric(openai_json["recording_time"]["start_date"]))
        end_date = str(convert_date_to_numeric(openai_json["recording_time"]["end_date"]))

    if max_height is None:
        max_height, unit = split_number_and_unit(openai_json["max_height_in_chart_background"])

    if days_recorded is None:
        print("days_recorded: ", days_recorded)
        days_recorded = openai_json["days_recorded"]

    hydroelectric_station = openai_json["hydroelectric_station"]

    # Initialize the number of weeks in the chart background, the maximum value, and the pixel distance between the numbers.
    week_line = int(days_recorded)
    max_scale = float(max_height)
    x_gap_h = float(interval)

    # Retrieve the image height information for subsequent calculations.
    img = Image.open(image_path)
    img_width, img_height = img.size

    if False:
        print("test05.py")
        # The logic of mutiple lines

    else:
        # print("openai json: ", openai_json)
        # print("interval: ", x_gap_h)
        # print("week_line: ", week_line)
        # print("max_scale: ", max_scale)

        # Retrieve the recorded time information.
        time_of_records = []
        for i in range(week_line):
            date_obj = datetime.strptime(start_date, '%d %b %Y')
            new_date_obj = date_obj + timedelta(days=i)
            new_date_str = str(new_date_obj.strftime('%d %b %Y'))
            for j in range(int(24/x_gap_h)):
                time_of_record = new_date_str + f" {x_gap_h*j} h"
                time_of_records.append(time_of_record)
        time_of_records.append(new_date_str + " 24 h")
        print("time recorded: ", time_of_records)

        # Calculate the distance between the numbers, which will later be adjusted based on the detected lines.
        gap = (abs(x2_frame - x1_frame)) / (week_line*(24/x_gap_h))

        # Obtain the actual coordinates of the grid lines along the x-axis.
        x_grid = []
        for i in range(int(week_line*(24/x_gap_h)+1)):
            x_grid.append(int(x1_frame + i*gap))

        # Adjust based on the actual detected lines: if the detected points deviate from the actual straight line, align them, allowing for a tolerance of half the value of the gap.
        detected_lines = detect_vertical_lines(image_path)
        tolerance = int(gap_interval/2)
        adjusted_x_grid = []
        for x in x_grid:
            adjusted_x = x
            for detected_x in detected_lines:
                if abs(x - detected_x) <= tolerance:
                    adjusted_x = detected_x
                    break
            adjusted_x_grid.append(adjusted_x)

        # Draw the adjusted vertical lines on the image for debugging purposes.
        img_draw = Image.open(image_path)  
        draw = ImageDraw.Draw(img_draw)
        img_draw_height = img_draw.size[1]     
        line_height_in_frame = []
        values = []
        feet_inches = []
        if adjust:
            x_grid_for = adjusted_x_grid
        else:
            x_grid_for = x_grid

        for x in x_grid_for:
            draw.line((x, 0, x, img_draw_height), fill="green", width=2) 
        img_draw = img_draw.convert("RGB")
        img_draw.save("output/image_with_lines.jpg")

        # Calculate the height of the polyline relative to the x-axis.
        for x in x_grid_for:
            if x in all_cols:
                y2_diff = img_height - y2_frame

                # Initialize the mean value.
                ave = None

                if (x-4) in all_cols and (x+4) in all_cols:
                    # If both `x-4` and `x+4` are in `all_cols`, calculate the mean of the values within the range from `x-4` to `x+4`.
                    points = [smoothed_points[all_cols.index(i)] for i in range(x-4, x+5)]
                    ave = sum([(img_height - point - y2_diff) for point in points]) / 9

                if (x-4) in all_cols and not (x+4) in all_cols:
                    # If only `x-4` is in `all_cols`, calculate the mean of the values within the range from `x` to `x-4`.
                    points = [smoothed_points[all_cols.index(i)] for i in range(x-4, x+1)]
                    ave = sum([(img_height - point - y2_diff) for point in points]) / 5

                if (x+4) in all_cols and not (x-4) in all_cols:
                    # If only `x+4` is in `all_cols`, calculate the mean of the values within the range from `x` to `x+4`.
                    points = [smoothed_points[all_cols.index(i)] for i in range(x, x+5)]
                    ave = sum([(img_height - point - y2_diff) for point in points]) / 5

                if ave is None:
                    # If none of the conditions are met, only process the current point.
                    ave = (img_height - smoothed_points[all_cols.index(x)] - y2_diff)

                line_height_in_frame.append(ave)
                # line_height_in_frame.append((img_height-smoothed_points[all_cols.index(x)])-(img_height-y2_frame))
            else:
                line_height_in_frame.append(None)

        # Calculate the value
        for line_height in line_height_in_frame:
            if line_height:
                values.append(max_scale*(line_height/(abs(y2_frame-y1_frame))))
            else:
                values.append(None)
        
        # Convert it into the representation in feet.
        for value in values:
            if value:
                foot, inch = decimal_to_feet_inches(value)
                feet_inches.append(f"{foot}feet {inch:.2f}inches")
            else:
                feet_inches.append(None)
        print("feet_inches: ", feet_inches)

        # Write the data into an Excel file
        df = pd.DataFrame({
            'Time': time_of_records,
            'Height': feet_inches
        })
        image_path_new = Path(image_path)
        new_image_path = image_path_new.parent.parent  # 获取上一级目录
        output_path = f'{new_image_path}/output/output.xlsx'
        df.to_excel(f'{output_path}', index=True)

# get_result(image_path="/Users/yuqiao/Desktop/project2/example4.png", days_recorded=7, interval=3)
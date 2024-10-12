import re
from collections import Counter
from scripts.microsoftapi import call_microsoft_api


def sample_interval(image_path):
    # Call the algorithms
    time_data, gap = call_microsoft_api(image_path=image_path)
    interval, time_format = detect_sampling_interval(time_data=time_data)
    return interval, time_format, gap

def detect_sampling_interval(time_data):
    # Step 1: Clean the data.
    cleaned_data = []
    for entry in time_data:
        # Use a regular expression to remove non-numeric characters and extract the numbers.
        numbers = re.findall(r'\d+', entry)
        for number in numbers:
            # Add the found numbers to the cleaned list.
            cleaned_data.append(number)
    
    # Step 2: Convert the cleaned numbers into a candidate time series.
    candidate_times = []
    for number in cleaned_data:
        # Handle potential time segmentation.
        if len(number) <= 2:
            candidate_times.append(int(number))
        else:
            # Long numbers need to be reasonably split into multiple time markers.
            for i in range(0, len(number), 2):
                candidate_times.append(int(number[i:i+2]))
    
    # Step 3: Determine the time format (12-hour or 24-hour system).
    count_1_to_12 = sum(1 for time in candidate_times if 1 <= time <= 12)
    count_13_to_24 = sum(1 for time in candidate_times if 13 <= time <= 24)
    
    # Use the distribution of numbers to statistically determine whether the time format is 12-hour or 24-hour.
    if count_13_to_24 > count_1_to_12 * 0.2:  # If the proportion of numbers between 13 and 24 exceeds 20%, it is more likely to be the 24-hour format.
        time_format = 24
    else:
        time_format = 12
    
    # Step 4: Infer Time Intervals
    # Calculate the difference between adjacent time points.
    time_diffs = []
    for i in range(1, len(candidate_times)):
        if time_format == 12:
            diff = (candidate_times[i] - candidate_times[i-1]) % 12
        else:
            diff = (candidate_times[i] - candidate_times[i-1]) % 24
        time_diffs.append(diff)
    
    # Calculate the most common time difference and use it as the estimated time interval.
    most_common_diff = Counter(time_diffs).most_common(1)[0][0]
    
    return most_common_diff, time_format





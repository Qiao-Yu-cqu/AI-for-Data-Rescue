import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from collections import Counter

def call_microsoft_api(image_path):
    """
    Calls the Microsoft Azure Computer Vision API to analyze an image and extract various visual features, 
    including tags, objects, captions, dense captions, people detection, smart crops, and text recognition.

    Args:
    image_path (str): The file path to the image that will be analyzed.

    Returns:
    list: A list of words detected in the image that have a confidence score greater than 0.5 and the text at the labeled positions on the x-axis..
    list: A list of bounding polygons corresponding to the detected words' positions in the image.

    Notes:
    - Requires environment variables 'VISION_ENDPOINT' and 'VISION_KEY' for authentication with Azure Computer Vision.
    - The function loads the image as a binary stream and passes it to the Azure API to analyze various visual features.
    - Specifically focuses on text detection (OCR) and returns detected words along with their bounding polygons.
    """

    # Set the values of your computer vision endpoint and computer vision key
    # as environment variables:
    try:
        os.environ['VISION_KEY'] = '0d7585cf40134217bfafc9719be44e70'
        os.environ['VISION_ENDPOINT'] = "https://computer-vision-api-qiao.cognitiveservices.azure.com/"
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Load image to analyze into a 'bytes' object
    with open(image_path, "rb") as f:
        image_data = f.read()

    visual_features =[
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.READ,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.PEOPLE,
        ]

    # Get a caption for the image. This will be a synchronously (blocking) call.
    # Analyze all visual features from an image stream. This will be a synchronously (blocking) call.
    result = client._analyze_from_image_data(
        image_data=image_data,
        visual_features=visual_features,
        smart_crops_aspect_ratios=[0.9, 1.33],
        gender_neutral_caption=True,
        language="en"
    )

    words = []
    boundings = []

    if result.read is not None:
        for line in result.read.blocks[0].lines:
            # print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
            for word in line.words:
                if word.confidence>0.5:
                    words.append(word.text)
                    boundings.append(word.bounding_polygon)
                    # print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")
    # print(words)
    # print("boundings: ", boundings)

    def find_modes(lst):

        # Find the most common one as output
        count = Counter(lst)
        
        most_common = count.most_common()
        mode = most_common[0][0]
        second_mode = None
        if len(most_common) > 1:
            second_mode = most_common[1][0]
        
        return mode, second_mode

    def filter_anomalies(gaps, tolerance=4):
        # Calculate the mode of all the intervals.
        mode = Counter(gaps).most_common(1)[0][0]
        
        # Filter out the values that are within the tolerance range of the mode.
        filtered_gaps = [gap for gap in gaps if abs(gap - mode) <= tolerance]

        return sum(filtered_gaps)/len(filtered_gaps)

    def find_x_axis_positions(words, boundings, tolerance=5):
        all_points_y_axis = []
        all_points_x_axis = []
        target_points_gap = []
        selected_words = []

        for i in range(len(words)):
            for points in boundings[i]:
                all_points_y_axis.append(points['y'])
                all_points_x_axis.append(points['x'])
        
        mode, second_mode = find_modes(all_points_y_axis)
        for i in range(len(words)-1):
            for points in boundings[i]:
                if ((mode-tolerance)<points['y']<(mode+tolerance)):
                    selected_words.append(words[i])
                    target_points_gap.append(abs(points['x']-boundings[i+1][1]['x']))
                    break
        return target_points_gap, selected_words


    target_points_gap, selected_words = find_x_axis_positions(words=words, boundings=boundings, tolerance=30)
    gap = filter_anomalies(target_points_gap)
    # print(words)
    return selected_words, gap

import base64
import requests
import re
import json

# OpenAI API Key
api_key = "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"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def call_openai_api(image_path):
    """
    Calls the OpenAI API to analyze an image of a water level chart and extract relevant information in JSON format.

    Args:
    image_path (str): The file path to the image that will be analyzed.

    Returns:
    dict: A JSON object containing the recording time (start and end date), 
          the maximum height in the chart background, the number of days recorded, 
          and the name of the hydroelectric station.

    Notes:
    - The function encodes the image as a base64 string and sends it to the OpenAI API 
      along with a request message asking for specific details about the chart.
    - The function uses the GPT-4 model to analyze the chart and return the relevant data.
    - The JSON response is parsed and returned if it is found and properly formatted. 
      Otherwise, an error message will be printed.
    - Requires an `api_key` for authentication with the OpenAI API, which must be set as a variable in the environment.
    - If the JSON content cannot be parsed, an error will be raised, and a message will be displayed.
    """

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": """This is an image showing the water level height changing over time. Please answer using a JSON format with the following details: the recording time of the image, the maximum height of the chart background, the number of days recorded, and the name of the hydroelectric station. The response format is as follows: \{ "recording_time": \{ "start_date": "xx (in numbers) xx (in numbers) xxxx", "end_date": "xx (in numbers) xx (in numbers) xxxx" \}, "max_height_in_chart_background": "x feet", "days_recorded": x, "hydroelectric_station": "xxxxxxxxxxxx" \}"""

            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    get_response = response.json()
    openai_str = get_response["choices"][0]["message"]["content"]

    match = re.search(r'\{[\s\S]*\}', openai_str)

    if match:
        json_data = match.group()
        # 将提取的字符串解析为JSON对象
        try:
            parsed_json = json.loads(json_data)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
    else:
        print("JSON content not found")

    
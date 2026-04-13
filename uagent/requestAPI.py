import base64
import io

import matplotlib.pyplot as plt
import requests
from PIL import Image

from config import (
    TOOL_SERVICE_DIOR_URL,
    TOOL_SERVICE_DOTA_URL,
    TOOL_SERVICE_LOVEDA_URL,
    TOOL_SERVICE_XVIEW_URL,
)


def _post_image(url, image_path):
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(url, files=files)
    response.raise_for_status()
    return response.json()


def requestAPI_xview(image_path):
    response_json = _post_image(TOOL_SERVICE_XVIEW_URL, image_path)
    return response_json["result"]


def requestAPI_DIOR(image_path):
    response_json = _post_image(TOOL_SERVICE_DIOR_URL, image_path)
    return response_json["result"]


def requestAPI_DOTA(image_path):
    response_json = _post_image(TOOL_SERVICE_DOTA_URL, image_path)
    try:
        return response_json
    except Exception:
        return "No object detected."


def requestAPI_loveda(image_path):
    response_json = _post_image(TOOL_SERVICE_LOVEDA_URL, image_path)
    print("???????:", response_json.get("message"))

    encoded_image_string = response_json.get("generated_image_png_base64")
    if encoded_image_string:
        image_raw_bytes = base64.b64decode(encoded_image_string)
        image_bytes_io = io.BytesIO(image_raw_bytes)
        received_image = Image.open(image_bytes_io)
        plt.imshow(received_image)
        plt.title("Received Generated Image")
        plt.axis("off")
        received_image.save("received_processed_image_base64.png")
    else:
        print("??????? Base64 ????????")

    return response_json

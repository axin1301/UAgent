"""
Model access utilities for the packaged UAgent directory.

This version prefers values from `config.py`, while still allowing overrides through
environment variables defined in that file.
"""

from openai import OpenAI
import base64

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    VLM_API_KEY,
    VLM_BASE_URL,
    VLM_MODEL,
)


def _build_vlm_client():
    return OpenAI(
        base_url=VLM_BASE_URL,
        api_key=VLM_API_KEY,
    )


def _build_llm_client():
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )


def VLM(paths, prompt):
    client = _build_vlm_client()

    base64_images = []
    image_paths = paths
    textprompt = prompt

    print('Image Paths: ', image_paths)
    try:
        for image_path in [image_paths]:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = "image/jpeg" if image_path.endswith(".jpg") else "image/png"
    base64_image_data = f"data:{mime_type};base64,{encoded_image}"
    base64_images.append(base64_image_data)

    content = []
    for item in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": item}
        })

    content.append({
        "type": "text",
        "text": str(textprompt)
    })

    response = client.chat.completions.create(
        extra_body={
            "options": {
                "num_ctx": 8192,
                "num_thread": 8,
            }
        },
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0.0,
        top_p=0.9,
        seed=42,
        max_tokens=4096,
    )
    return response.choices[0].message.content, 0, 0, 0


def LLM(prompt):
    client = _build_llm_client()
    dialogs = [{
        "role": "user",
        "content": prompt,
    }]

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=dialogs,
        temperature=0.0,
        top_p=0.9,
        max_tokens=4096,
    )

    return completion.choices[0].message.content

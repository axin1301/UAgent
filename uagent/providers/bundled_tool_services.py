from config import (
    TOOL_SERVICE_DIOR_URL,
    TOOL_SERVICE_DOTA_URL,
    TOOL_SERVICE_LOVEDA_URL,
    TOOL_SERVICE_XVIEW_URL,
)
from providers.bundled_tool_paths import get_bundled_tool_script


BUNDLED_TOOL_SERVICES = {
    "dior": {
        "url": TOOL_SERVICE_DIOR_URL,
        "script": get_bundled_tool_script("mmdet.dior"),
        "port": 8001,
        "endpoint": "/inference_DIOR_one_image",
    },
    "xview": {
        "url": TOOL_SERVICE_XVIEW_URL,
        "script": get_bundled_tool_script("mmdet.xview"),
        "port": 8002,
        "endpoint": "/inference_xview_one_image",
    },
    "dota": {
        "url": TOOL_SERVICE_DOTA_URL,
        "script": get_bundled_tool_script("ultralytics.dota_final"),
        "port": 8003,
        "endpoint": "/inference_DOTA_one_image",
    },
    "loveda": {
        "url": TOOL_SERVICE_LOVEDA_URL,
        "script": get_bundled_tool_script("mmseg.loveda_msi"),
        "port": 8004,
        "endpoint": "/inference_loveda_one_image",
    },
}


def get_bundled_tool_service(name: str):
    if name not in BUNDLED_TOOL_SERVICES:
        raise KeyError(f"Unknown bundled tool service: {name}")
    return BUNDLED_TOOL_SERVICES[name]

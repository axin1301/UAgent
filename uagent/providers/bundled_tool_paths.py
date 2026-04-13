import os

from config import THIRD_PARTY_TOOLS_ROOT


def _tool_path(*parts: str) -> str:
    return os.path.join(THIRD_PARTY_TOOLS_ROOT, *parts)


BUNDLED_TOOL_SCRIPTS = {
    "ultralytics.dota": _tool_path("ultralytics", "inference_DOTA_one_image.py"),
    "ultralytics.dota_final": _tool_path("ultralytics", "inference_DOTA_one_image_final.py"),
    "ultralytics.dota_msi": _tool_path("ultralytics", "inference_DOTA_one_image_msi.py"),
    "mmdet.dior": _tool_path("mmdetection", "inference_DIOR_one_image.py"),
    "mmdet.generic": _tool_path("mmdetection", "inference_one_image.py"),
    "mmdet.xview": _tool_path("mmdetection", "inference_xview_one_image.py"),
    "mmdet.xview_msi": _tool_path("mmdetection", "inference_xview_one_image_msi.py"),
    "mmseg.loveda": _tool_path("mmsegmentation", "inference_loveda_one_image.py"),
    "mmseg.loveda_msi": _tool_path("mmsegmentation", "inference_loveda_one_image_msi.py"),
    "mmseg.seg": _tool_path("mmsegmentation", "inference_seg_one_image.py"),
    "mmseg.seg_api": _tool_path("mmsegmentation", "inference_seg_one_image_API.py"),
}


def get_bundled_tool_script(name: str) -> str:
    if name not in BUNDLED_TOOL_SCRIPTS:
        raise KeyError(f"Unknown bundled tool script: {name}")
    return BUNDLED_TOOL_SCRIPTS[name]

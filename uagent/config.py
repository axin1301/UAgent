"""
Central configuration for the packaged UAgent directory.

Update these values before public release or production use.
"""

import os


# Text model endpoint
LLM_BASE_URL = os.getenv("UAGENT_LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("UAGENT_LLM_API_KEY", "")
LLM_MODEL = os.getenv("UAGENT_LLM_MODEL", "qwen/qwen-2.5-7b-instruct")

# Vision-language model endpoint
VLM_BASE_URL = os.getenv("UAGENT_VLM_BASE_URL", "http://localhost:11434/v1")
VLM_API_KEY = os.getenv("UAGENT_VLM_API_KEY", "ollama")
VLM_MODEL = os.getenv("UAGENT_VLM_MODEL", "qwen2.5vl:7b")

# Optional defaults for packaged batch runs
DEFAULT_TASK = os.getenv("UAGENT_DEFAULT_TASK", "all_global_gdp_task")
DEFAULT_START = int(os.getenv("UAGENT_DEFAULT_START", "10"))
DEFAULT_END = int(os.getenv("UAGENT_DEFAULT_END", "60"))

# Bundled third-party tool scripts
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
UAGENT_ROOT = os.path.dirname(PACKAGE_ROOT)
THIRD_PARTY_TOOLS_ROOT = os.getenv(
    "UAGENT_THIRD_PARTY_TOOLS_ROOT",
    os.path.join(UAGENT_ROOT, "third_party_tools"),
)
WEIGHTS_ROOT = os.getenv(
    "UAGENT_WEIGHTS_ROOT",
    os.path.join(UAGENT_ROOT, "weights"),
)

# Local bundled tool service endpoints
TOOL_SERVICE_DIOR_URL = os.getenv("UAGENT_TOOL_SERVICE_DIOR_URL", "http://127.0.0.1:8001/inference_DIOR_one_image")
TOOL_SERVICE_XVIEW_URL = os.getenv("UAGENT_TOOL_SERVICE_XVIEW_URL", "http://127.0.0.1:8002/inference_xview_one_image")
TOOL_SERVICE_DOTA_URL = os.getenv("UAGENT_TOOL_SERVICE_DOTA_URL", "http://127.0.0.1:8003/inference_DOTA_one_image")
TOOL_SERVICE_LOVEDA_URL = os.getenv("UAGENT_TOOL_SERVICE_LOVEDA_URL", "http://127.0.0.1:8004/inference_loveda_one_image")

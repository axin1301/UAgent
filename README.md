# UAgent (UAgent4Zero-shot-socioeconomic-prediction)

We present UAgent, an evidence-constrained multimodal agent framework for zero-shot socioeconomic inference from urban visual data. UAgent reformulates prediction as a structured, evidence-driven reasoning process that decomposes socioeconomic queries into modality-aware visual signals, constructs an explicit urban state grounded in tool-derived features, and incorporates iterative self-reflection to detect missing or inconsistent evidence before producing final estimates. By constraining inference to structured visual evidence, UAgent improves interpretability and calibration while mitigating unsupported predictions.

![](figs/UAgent-Framework.png)

This directory packages the core pipeline from the research codebase into a more reusable and GitHub-friendly layout.

## Overview

The pipeline is designed around a staged reasoning process:

1. `Initialization`
2. `Analysis`
3. `Tool-grounded evidence collection`
4. `Urban-state construction`
5. `Reflection`
6. `Reasoning and conclusion`

## Features

- Evidence-grounded multimodal reasoning over satellite and street-view imagery
- Structured agent pipeline instead of direct one-shot prediction
- Tool-based evidence collection and urban-state construction
- Reflection stage for exposing weak or missing evidence
- Packaged external-tool interface for integrating your own domain tools
- Bundled third-party inference entry scripts under `third_party_tools/`
- Unified `weights/` directory for third-party checkpoints
- Generic dataset runner based on JSON input plus an image directory

## Repository Layout

```text
UAgent/
|- README.md
|- environment.yml
|- requirements.txt
|- requirements-agent.txt
|- requirement-tool.txt
|- docs/
|  |- external_tools_api.md
|  `- start_tools.md
|- examples/
|  |- demo_single_run.py
|- figs/
|  `- UAgent-Framework.png
|- scripts/
|  |- start_tools.ps1
|  `- start_tools.bat
|- weights/
|  `- README.md
|- third_party_tools/
|  |- README.md
|  |- common_paths.py
|  |- support/
|  |- ultralytics/
|  |- mmdetection/
|  `- mmsegmentation/
`- uagent/
   |- __init__.py
   |- config.py
   |- run_pipeline.py
   |- run_dataset.py
   |- external_tools_interface.py
   |- providers/
   |  |- __init__.py
   |  |- bundled_tool_paths.py
   |  `- bundled_tool_services.py
   |- func_sup.py
   |- prompt_list.py
   |- llm_api.py
   |- tool_list_short.py
   |- tool_function_map.py
   |- Tools_def.py
   |- requestAPI.py
   `- ...
```

## Expected Input Format

The packaged runner assumes:

- one JSON file containing a list of samples
- one image directory containing all referenced images

Each sample should look like:

```json
{
  "id": "sample_001",
  "images": [
    "sat_001.png",
    "stv_001.jpg",
    "stv_002.jpg"
  ],
  "prompt": "Based on the provided satellite image and street view photos, estimate the target indicator.",
  "reference_normalized": 6.4
}
```

Notes:

- `images` may contain relative paths or file names; they will be resolved relative to `--image-dir`
- `prompt` or `text` is required
- `reference_normalized` is optional and only needed if you want to evaluate predictions later
- if `--image-dir` is omitted, the runner defaults to an `images/` directory next to the input JSON

## Core Files

- `uagent/func_sup.py`: core pipeline logic and agent stages
- `uagent/prompt_list.py`: prompt templates for each stage
- `uagent/llm_api.py`: text and multimodal model access
- `uagent/tool_function_map.py`: mapping from tool names to implementations
- `uagent/Tools_def.py`: built-in tool definitions
- `uagent/external_tools_interface.py`: clean interface for swapping in external tool providers
- `uagent/providers/bundled_tool_paths.py`: local lookup table for bundled third-party scripts
- `uagent/providers/bundled_tool_services.py`: centralized bundled service registry
- `uagent/requestAPI.py`: HTTP client layer used by the current tool wrappers

## Environment Setup

Recommended Python version:

- `Python 3.8.16`

The packaged repository now includes:

- `requirements.txt`: merged environment for the agent pipeline and bundled tool services
- `environment.yml`: conda environment file for the same packaged setup

The original split files are kept for reference:

- `requirements-agent.txt`
- `requirement-tool.txt`

### Option A: Conda via `environment.yml`

```bash
conda env create -f environment.yml
conda activate uagent
```

This is the simplest recommended setup if you use conda.

### Option B: Manual conda setup

```bash
conda create -n uagent python=3.8.16 -y
conda activate uagent
```

Install PyTorch first. Choose the build that matches your machine and CUDA setup.
For example, for CUDA 11.8:

```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:

```bash
pip install torch==2.0.0 torchvision==0.15.0
```

Then install the merged dependencies:

```bash
pip install -r requirements.txt
```

### Option C: venv

```bash
python -m venv .venv
.venv\Scripts\activate
```

Then install PyTorch and the merged dependencies as above.

### Notes on installation

- `mmcv`, `mmengine`, `mmsegmentation`, `ultralytics`, and related packages are included in the merged environment because the bundled third-party tools depend on them.
- Some tool services may still require additional system-level or CUDA-specific setup depending on your machine.
- If you already maintain separate environments for agent and tool services, you may continue using the original split requirement files instead of the merged one.

## Setup

### 1. Download checkpoints into `weights/`

Place all required third-party model weights into `UAgent/weights/`.

Expected filenames:

- `yolo11l-obb.pt`
- `dior-rvsa-b-mae-mtp-epoch_12.pth`
- `xview-rvsa-l-mae-mtp_epoch_12.pth`
- `loveda-rvsa-b-mae-mtp-iter_80000.pth`

### 2. Configure language / vision-language models

By default, the packaged code expects:

- a local multimodal endpoint for `VLM`
- a text model endpoint for `LLM`

See `uagent/config.py` and `uagent/llm_api.py` for details.

### 3. Prepare your dataset

Minimal directory layout:

```text
my_dataset/
|- data.json
`- images/
   |- sat_001.png
   |- stv_001.jpg
   `- stv_002.jpg
```

## Overall Run Procedure

The current packaged tool chain still follows a service-style setup for several bundled third-party tools.

The intended full workflow is:

1. create the Python 3.8.16 environment and install dependencies
2. place all checkpoints into `UAgent/weights/`
3. start the bundled inference services you need
4. run the main UAgent pipeline on your dataset

## Start Bundled Tool Services

The default bundled service endpoints are:

- `8001`: DIOR detection
- `8002`: xView detection
- `8003`: DOTA detection
- `8004`: LoveDA segmentation

These URLs are centralized in `uagent/config.py`.

Example startup commands from the `UAgent/` directory:

```bash
python third_party_tools/mmdetection/inference_DIOR_one_image.py
python third_party_tools/mmdetection/inference_xview_one_image.py
python third_party_tools/ultralytics/inference_DOTA_one_image_final.py
python third_party_tools/mmsegmentation/inference_loveda_one_image_msi.py
```

Or use one of the helpers:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tools.ps1
```

```bat
scripts\start_tools.bat
```

More details:

- `docs/start_tools.md`

Notes:

- you do not necessarily need to start every service if your selected tool chain does not use all of them
- for the current default satellite object-detection path, DIOR + xView + DOTA are all referenced
- for the current default satellite semantic-segmentation path, the LoveDA segmentation service is referenced

## Run UAgent

Minimal usage:

```bash
python uagent/run_dataset.py --input-json /path/to/data.json
```

This will automatically assume:

- images live in `/path/to/images/`
- predictions are saved to `/path/to/data_uagent_predictions.json`

If your images are stored elsewhere, specify them explicitly:

```bash
python uagent/run_dataset.py \
  --input-json /path/to/data.json \
  --image-dir /path/to/all_images
```

You can also control output path, slicing, and worker count:

```bash
python uagent/run_dataset.py \
  --input-json /path/to/data.json \
  --image-dir /path/to/all_images \
  --output-json /path/to/predictions.json \
  --start 0 \
  --end 100 \
  --workers 4
```

Minimal packaged entry point:

```bash
python -m uagent.run_pipeline
```

This runner is intentionally lightweight and mainly serves as a clean package-level entry point.

## Current Tool Chain

The current packaged tool chain is organized as:

1. `uagent/tool_function_map.py`
2. `uagent/Tools_def.py`
3. `uagent/requestAPI.py`
4. `uagent/providers/bundled_tool_services.py`
5. `third_party_tools/*/inference_*.py`

For the bundled satellite detector/segmenter path, the intended flow is:

- `Satellite Image Object Detection Tool`
- `Tools_def.py -> Object_Detection_Sat(...)`
- `requestAPI_DIOR / requestAPI_DOTA / requestAPI_xview`
- bundled local service endpoints defined in `config.py`
- bundled inference scripts under `third_party_tools/`

For the bundled satellite segmentation path, the intended flow is:

- `Satellite Image Semantic Segmentation Tool`
- `Tools_def.py -> requestAPI_loveda(...)`
- bundled local service endpoint defined in `config.py`
- bundled LoveDA segmentation script under `third_party_tools/mmsegmentation/`


## Street-View Tool Status

The current street-view toolchain is not yet fully unified in the same way as the bundled satellite toolchain.

At the moment, street-view tools fall into three categories:

1. `VLM-based tools`

These tools directly call the multimodal model through `VLM(image_path, prompt)` and do not depend on precomputed external result files.
Examples include:

- `Text Sign OCR`
- `Street View Image Captioner`
- `Commercial Clue Extractor`
- several other street-view reasoning tools in `uagent/Tools_def.py`

2. `Precomputed-file / precomputed-metadata tools`

Some street-view tools currently read pre-generated results instead of invoking a bundled third-party service at runtime.
The main examples are:

- `Street_View_Semantic_Segmentation_Tool`
  - reads precomputed street-view segmentation outputs from paths such as `../ImageData/stv_semseg_result/` or `D:/Citylens_image_part/stv_semseg_result_Citylens/`
- `Street_Object_Detector`
  - reads precomputed detection information from in-memory metadata structures such as `data_stv_info`, and may also fall back to `my_data.json`

3. `Not-yet-fully-refactored tools`

Some street-view modules still reflect the original research setup and have not yet been migrated into the bundled `third_party_tools/` service structure.

### Practical implication

The packaged repository is currently most self-consistent for:

- the main UAgent pipeline
- bundled satellite detection/segmentation services
- VLM-based street-view reasoning tools

For street-view segmentation and street-view object detection, the current code still assumes access to precomputed intermediate artifacts unless you further refactor those modules.

If you plan to publish this repository, the safest description is:

- satellite external tools are partially bundled and centralized
- street-view tools are partly VLM-based and partly dependent on precomputed outputs from the original research workflow
## Bundled Third-Party Tools

To keep the GitHub package self-contained, the external inference entry scripts used in the original research environment are copied into:

- `third_party_tools/ultralytics/`
- `third_party_tools/mmdetection/`
- `third_party_tools/mmsegmentation/`

These scripts now look up checkpoints from the unified `weights/` directory and use bundled support files under `third_party_tools/support/`.

Important note:

- these bundled scripts may still depend on external framework packages such as Ultralytics, MMDetection, MMSegmentation, MMEngine, MMCV, and related dependencies
- bundling the scripts, configs, and support files centralizes the repository layout, but some tools may still require follow-up environment setup

## External Tool Interface

External integrations are separated into:

- `uagent/external_tools_interface.py`

See:

- `docs/external_tools_api.md`

## Using Your Own Tools

The intended integration pattern is:

1. implement a provider class that follows the interface in `external_tools_interface.py`
2. expose each tool under a stable tool name
3. connect those tool names to the pipeline through `tool_function_map.py` or your own adapter layer

Minimal tool contract:

```python
result = provider.run_tool(
    tool_name="Road Network Extractor",
    image_path="/abs/path/to/image.png",
    prompt="Describe the road layout and connectivity."
)
```

Expected returned structure:

```python
{
    "tool_name": "Road Network Extractor",
    "status": "ok",
    "output": "...",
    "metadata": {}
}
```

## What Is Preserved From the Research Code

- the original multi-stage agent pipeline
- the original prompt templates
- the built-in tool mapping and tool definitions
- the original LLM/VLM access pattern

## What Is Intentionally Simplified Here

- experiment orchestration remains in the original project root
- dataset-specific benchmark scripts are not duplicated inside `UAgent/`
- external domain tools are exposed through a cleaner interface rather than fully refactored

## Reproducibility Notes

- The packaged directory is designed for readability and reuse.
- Exact paper reproduction should still use the original root-level experiment scripts and evaluation scripts.
- The packaged interface is best viewed as a release-oriented wrapper around the research implementation.

## Notes

- This packaging keeps the original research implementation style, so some files still reflect experiment-oriented assumptions.
- The original root-level scripts are not modified.
- The packaged directory is meant to make the pipeline easier to share, document, and refactor incrementally.

## Citation

If you reuse this packaged directory in a paper or repository, describe it as:

> a packaged version of the UAgent evidence-grounded multimodal urban reasoning pipeline, with an explicit external tool interface for domain-tool integration



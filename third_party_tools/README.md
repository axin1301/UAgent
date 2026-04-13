# Bundled Third-Party Tool Entrypoints

This directory contains the inference entry scripts copied into the packaged `UAgent` repository so that tool invocation can be organized from within the same GitHub project.

## Structure

- `ultralytics/`: bundled YOLO/OBB inference entry scripts
- `mmdetection/`: bundled detection entry scripts for DIOR, xView, and related detectors
- `mmsegmentation/`: bundled semantic-segmentation inference entry scripts
- `support/`: copied configs, backbone files, and dataset registration helpers needed by the bundled scripts
- `common_paths.py`: shared path helpers for weights and bundled support files

## Unified Weights Directory

All bundled third-party scripts are being normalized to read checkpoints from:

- `UAgent/weights/`

Expected filenames:

- `yolo11l-obb.pt`
- `dior-rvsa-b-mae-mtp-epoch_12.pth`
- `xview-rvsa-l-mae-mtp_epoch_12.pth`
- `loveda-rvsa-b-mae-mtp-iter_80000.pth`

## Intended Usage

The recommended integration pattern is:

1. keep the agent pipeline in `uagent/`
2. keep external-model entry scripts in `third_party_tools/`
3. download all checkpoints into `UAgent/weights/`
4. add thin provider/wrapper layers in `uagent/` that invoke these local scripts

## Important Note

These scripts are now centralized inside `UAgent/`, but some of them may still depend on external Python packages and further environment setup. Bundling the scripts, configs, and support files makes the repository more self-contained, but it does not guarantee that every third-party tool is fully portable without its corresponding framework dependencies.

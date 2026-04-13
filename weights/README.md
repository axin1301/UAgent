# Unified Weights Directory

Place all model checkpoints used by bundled third-party tools into this directory.

## Expected filenames

- `yolo11l-obb.pt`
- `dior-rvsa-b-mae-mtp-epoch_12.pth`
- `xview-rvsa-l-mae-mtp_epoch_12.pth`
- `loveda-rvsa-b-mae-mtp-iter_80000.pth`

The bundled inference scripts are being normalized to look up checkpoints from `UAgent/weights/` so users can download all required weights into a single location before running the pipeline.

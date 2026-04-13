# Start Bundled Tool Services

From the `UAgent/` directory, the default bundled third-party services can be started as follows.

## Default services

- `8001`: DIOR detection
- `8002`: xView detection
- `8003`: DOTA detection
- `8004`: LoveDA segmentation

## Manual startup commands

```bash
python third_party_tools/mmdetection/inference_DIOR_one_image.py
python third_party_tools/mmdetection/inference_xview_one_image.py
python third_party_tools/ultralytics/inference_DOTA_one_image_final.py
python third_party_tools/mmsegmentation/inference_loveda_one_image_msi.py
```

## PowerShell helper

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tools.ps1
```

## Batch helper

```bat
scripts\start_tools.bat
```

This opens one terminal window per bundled service.

## Notes

- Make sure the required framework dependencies are installed before starting the services.
- Make sure all required checkpoints are placed in `UAgent/weights/`.
- You do not necessarily need every service for every experiment, but the current default satellite object-detection path references DIOR, xView, and DOTA, while the default satellite semantic-segmentation path references LoveDA.

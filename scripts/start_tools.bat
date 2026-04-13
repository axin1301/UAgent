@echo off
set SCRIPT_DIR=%~dp0
set UAGENT_ROOT=%SCRIPT_DIR%..

start "DIOR" powershell -NoExit -Command "Set-Location '%UAGENT_ROOT%'; python third_party_tools/mmdetection/inference_DIOR_one_image.py"
start "xView" powershell -NoExit -Command "Set-Location '%UAGENT_ROOT%'; python third_party_tools/mmdetection/inference_xview_one_image.py"
start "DOTA" powershell -NoExit -Command "Set-Location '%UAGENT_ROOT%'; python third_party_tools/ultralytics/inference_DOTA_one_image_final.py"
start "LoveDA" powershell -NoExit -Command "Set-Location '%UAGENT_ROOT%'; python third_party_tools/mmsegmentation/inference_loveda_one_image_msi.py"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$uagentRoot = Split-Path -Parent $root

$commands = @(
    'Set-Location "' + $uagentRoot + '"; python third_party_tools/mmdetection/inference_DIOR_one_image.py',
    'Set-Location "' + $uagentRoot + '"; python third_party_tools/mmdetection/inference_xview_one_image.py',
    'Set-Location "' + $uagentRoot + '"; python third_party_tools/ultralytics/inference_DOTA_one_image_final.py',
    'Set-Location "' + $uagentRoot + '"; python third_party_tools/mmsegmentation/inference_loveda_one_image_msi.py'
)

foreach ($cmd in $commands) {
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $cmd
}

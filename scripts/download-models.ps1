# Downloads the two model files into .\models (run from the project folder).
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
Invoke-WebRequest -Uri "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" -OutFile "models\emotion-ferplus-8.onnx"
Get-ChildItem models | Select-Object Name, Length

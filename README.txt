DROPO RunPod Serverless worker

Files:
- handler.py: Serverless version of the current Auto Fit logic.
- requirements.txt: Python dependencies.
- Dockerfile: GPU worker image.
- .dockerignore

Important:
The worker expects Queue endpoint JSON:
{
  "input": {
    "image_base64": "<base64 image>"
  }
}

It returns JSON with image_base64 containing the finished JPEG.

This keeps the current Dropo processing:
- BiRefNet General
- 2000x2000
- #EDF1EF background
- alpha MinFilter(5)
- NORMAL / LONG auto detection at ratio >= 1.65

Next step:
Put these files in a GitHub repository or build/push the Docker image to a registry.

# IDOL

IDOL is a FastAPI web app for cutting idol dance videos into loopable practice sections.

Upload a dance video, and the app detects beats and body keypoints, groups similar bone movements into practice parts, and lets you loop each part from a simple video UI.

## Features

- Upload a local dance video
- Automatic pose tracking with YOLO pose models
- Beat-aware section generation based on similar bone movement
- Initial section names generated from representative frames with an OpenAI vision model
- Loop playback by section pins on the timeline
- Playback speed and mirror mode
- Normal / rough part granularity controls
- Simple UI focused on practice, with detailed count/metronome information hidden

## Requirements

- Python 3.10 or newer
- ffmpeg
- A machine that can run Ultralytics YOLO pose inference

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

To enable OpenAI-generated initial section names, set an API key before running:

```bash
$env:OPENAI_API_KEY="your_api_key_here"
```

The default vision model is `gpt-5.4-mini`. You can override it with:

```bash
$env:OPENAI_CHUNK_NAME_MODEL="gpt-5.4-mini"
```

Image input detail defaults to `high` for better choreography naming. You can lower it with:

```bash
$env:OPENAI_CHUNK_NAME_IMAGE_DETAIL="low"
```

If `OPENAI_API_KEY` is not set, the app still analyzes videos normally and leaves section names empty until the user edits them.

On macOS/Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install ffmpeg separately if it is not already available.

Windows:

```bash
winget install Gyan.FFmpeg
```

macOS:

```bash
brew install ffmpeg
```

## Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

- Generated analysis files are stored under `work/` and are not meant to be committed.
- YOLO model weights (`*.pt`) are ignored by Git. Ultralytics can download the configured model when needed.
- Start with short videos while testing, because pose inference can take time.

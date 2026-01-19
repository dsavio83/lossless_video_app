# Lossless Video Processor

A full-stack web application for **lossless** video frame extraction and rendering using Python, FastAPI, and FFmpeg.

## Features
- **Lossless Extraction:** Converts video frames to PNG images (lossless compression) without color subsampling or artifacts.
- **Accurate Rendering:** Reassembles frames into video preserving original framerate and using lossless H.264 (High 4:4:4 Predictive profile) or RGB mode.
- **Simple UI:** Clean web interface for uploading and processing.

## Prerequisites

1. **Python 3.8+** installed.
2. **FFmpeg** installed and added to your system PATH.
   - **Windows:** 
     1. Download `ffmpeg-release-essentials.zip` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
     2. Extract it.
     3. Copy `bin/ffmpeg.exe` and `bin/ffprobe.exe` to a folder in your PATH (e.g., `C:\Windows\System32`) OR add the extracted `bin` folder to your User Environment Variables.
   - **Verify:** Open a new terminal and type `ffmpeg -version`.

## Installation

1. Navigate to the project directory:
   ```bash
   cd lossless_video_app
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the server:
   ```bash
   python main.py
   ```
   Or explicitly with uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

2. Open your browser and go to:
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

## usage

1. **Extract Frames:**
   - Click "Choose File" and select your video.
   - Click "Extract Frames". 
   - Wait for the "Success" message. A preview of the first frame will appear.

2. **Render Video:**
   - Once extraction is complete, the "Render to Video" button enables.
   - Click it to re-assemble the PNGs into `rendered_lossless.mp4`.
   - The file will automatically download.

## Technical Details

- **Extraction:**
  - `ffmpeg -i input.mp4 -vsync 0 extracted_frames/frame_%04d.png`
  - Uses PNG for mathematically lossless storage.
  - `-vsync 0` prevents frame dropping/duplication.

- **Rendering:**
  - `ffmpeg -framerate <fps> -i extracted_frames/frame_%04d.png -c:v libx264rgb -crf 0 -preset veryslow rendered_video/output.mp4`
  - `libx264rgb`: Ensures RGB color space is preserved (no YUV conversion loss).
  - `-crf 0`: Constant Rate Factor 0 triggers lossless mode in x264.

import os
import shutil
import subprocess
import json
import time
import asyncio
import logging
import string
import platform
from typing import List, Optional
from pathlib import Path
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import cv2
import cv2
import numpy as np

# Installer Module
import installer


# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
INPUT_DIR = BASE_DIR / "input_video"
OUTPUT_DIR = BASE_DIR / "extracted_frames"
TEMP_DIR = BASE_DIR / "temp"
RENDER_DIR = BASE_DIR / "rendered_video"

# Ensure directories
for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, RENDER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/frames", StaticFiles(directory=OUTPUT_DIR), name="frames")

# --- Binary Resolution ---
FFMPEG_DIR = BASE_DIR / "ffmpeg"

def resolve_binary(binary_name):
    """Finds ffmpeg/ffprobe in system path or project folders."""
    # 1. Check local Install dir first (priority)
    local_bin = installer.is_ffmpeg_installed(FFMPEG_DIR)
    if local_bin:
        # If looking for ffprobe and we found ffmpeg (or vice versa), 
        # usually they are in the same folder.
        bin_dir = Path(local_bin).parent
        target = bin_dir / (binary_name + (".exe" if os.name == 'nt' else ""))
        if target.exists():
            return str(target)
            
    # 2. Check System Path
    if shutil.which(binary_name):
        return shutil.which(binary_name)
    
    # 3. Fallbacks
    return binary_name

FFPROBE_BIN = resolve_binary("ffprobe")
FFMPEG_BIN = resolve_binary("ffmpeg")

# --- Helper Functions ---
def get_video_info(video_path: Path):
    """Returns (fps, duration, width, height) using ffprobe."""
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height,duration",
            "-of", "json", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        
        # Parse FPS
        fps_str = stream.get("r_frame_rate", "30/1")
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den else 30.0
        else:
            fps = float(fps_str)
            
        return {
            "fps": fps,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": float(stream.get("duration", 0))
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {"fps": 30.0, "width": 0, "height": 0, "duration": 0}

def format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_status(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                pass

manager = ConnectionManager()

# --- Global State for Processing Control ---
class ProcessingState:
    is_running = False
    should_stop = False
    
    # Setup State
    install_status = "idle" # idle, downloading, extracting, error, complete
    install_progress = 0
    install_msg = ""

process_state = ProcessingState()

# --- Core Processing Logic ---
async def process_video_task(video_path_str: str, threshold: float):
    process_state.is_running = True
    process_state.should_stop = False
    
    video_path = Path(video_path_str)
    
    # Clean output directory
    for f in OUTPUT_DIR.glob("*"):
        try: f.unlink() 
        except: pass

    try:
        # Get Info
        info = get_video_info(video_path)
        fps = info["fps"]
        total_duration = info["duration"]
        
        await manager.broadcast_status({
            "type": "start",
            "fps": fps,
            "total_duration": total_duration
        })

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Could not open video file.")

        frame_count = 0
        saved_count = 0
        segments = [] # Metadata for rendering
        
        prev_hash = None
        last_saved_frame_idx = 0
        
        # Buffer for the 'current unique frame' to save it when we know its duration
        # Structure: (frame_image, frame_index, start_timestamp)
        pending_segment = None

        while True:
            if process_state.should_stop:
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Resize for comparison speed (64x64 grayscale)
            small = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            # 2. Check difference
            is_unique = False
            if prev_hash is None:
                is_unique = True
            else:
                # Calculate absolute difference
                diff = cv2.absdiff(prev_hash, gray)
                score = np.sum(diff)
                if score > threshold:
                    is_unique = True
            
            current_time = frame_count / fps

            if is_unique:
                # A new scene has started.
                # A. Finalize the *previous* segment if it exists
                if pending_segment:
                    p_img, p_idx, p_start = pending_segment
                    p_end = current_time
                    p_dur = p_end - p_start
                    
                    filename = f"frame_{p_idx:06d}.png"
                    out_path = OUTPUT_DIR / filename
                    cv2.imwrite(str(out_path), p_img)
                    
                    saved_count += 1
                    await manager.broadcast_status({
                        "type": "new_frame",
                        "frame": {
                            "filename": filename,
                            "url": f"/frames/{filename}",
                            "start": format_time(p_start),
                            "end": format_time(p_end),
                            "duration": f"{p_dur:.2f}s"
                        }
                    })
                    
                    segments.append({
                        "file": filename,
                        "duration": p_dur
                    })

                # B. Start tracking the *new* segment
                pending_segment = (frame.copy(), frame_count, current_time)
                prev_hash = gray
            
            frame_count += 1
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                await manager.broadcast_status({
                    "type": "progress",
                    "progress": (current_time / total_duration) * 100 if total_duration else 0,
                    "current_time": format_time(current_time)
                })
            
            # CRITICAL: Yield control to event loop so WebSockets don't time out
            await asyncio.sleep(0)

        # End of video: Save the last pending segment
        if pending_segment:
            p_img, p_idx, p_start = pending_segment
            p_end = frame_count / fps
            p_dur = p_end - p_start
            
            filename = f"frame_{p_idx:06d}.png"
            out_path = OUTPUT_DIR / filename
            cv2.imwrite(str(out_path), p_img)
            
            await manager.broadcast_status({
                "type": "new_frame",
                "frame": {
                    "filename": filename,
                    "url": f"/frames/{filename}",
                    "start": format_time(p_start),
                    "end": format_time(p_end),
                    "duration": f"{p_dur:.2f}s"
                }
            })

        cap.release()
        
        # Save segments metadata with source video path
        metadata = {
            "source_video": str(video_path),
            "original_name": video_path.name,
            "fps": fps,
            "segments": segments
        }
        with open(OUTPUT_DIR / "segments.json", "w") as f:
            json.dump(metadata, f)
            
        # Legacy/Simple metadata.json for quick info
        simple_meta = {
            "fps": fps,
            "original_name": video_path.name,
            "source_path": str(video_path),
            "frame_count": saved_count
        }
        with open(BASE_DIR / "metadata.json", "w") as f:
            json.dump(simple_meta, f)
            
        await manager.broadcast_status({"type": "complete", "subtype": "extraction", "count": saved_count})

    except Exception as e:
        logger.error(f"Processing error: {e}")
        await manager.broadcast_status({"type": "error", "message": str(e)})
    finally:
        process_state.is_running = False

@app.post("/api/open")
def open_file(item: dict):
    # Security: Ensure we only open files in valid directories
    filename = item.get("filename")
    if not filename:
        return {"error": "No filename"}
    
    # Try finding in outputs first
    target = OUTPUT_DIR / filename
    if not target.exists():
        # Maybe it's a full path?
        try:
            p = Path(filename)
            if p.exists():
                target = p
        except:
            pass
            
    if not target.exists():
        return {"error": "File not found"}
        
    try:
        if platform.system() == "Windows":
            # Use standard startfile which usually brings window to front
            os.startfile(str(target))
            
        elif platform.system() == "Darwin":
            subprocess.call(["open", str(target)])
        else:
            subprocess.call(["xdg-open", str(target)])
        return {"status": "opened"}
    except Exception as e:
        return {"error": str(e)}

# --- Routes ---

@app.get("/")
def index():
    return FileResponse(APP_DIR / "static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(
        content=base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="),
        media_type="image/png"
    )

@app.get("/api/videos")
def list_videos(path: Optional[str] = None):
    target = Path(path) if path else INPUT_DIR
    if not target.exists():
        target = BASE_DIR # Fallback
        
    videos = []
    try:
        for f in target.iterdir():
            if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                videos.append(f.name)
        return {"path": str(target), "videos": sorted(videos)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/browse")
def browse_filesystem(path: Optional[str] = None):
    try:
        if not path:
            # List drives on Windows
            drives = [f"{d}:/" for d in string.ascii_uppercase if os.path.exists(f"{d}:")]
            return {"current": "", "directories": drives, "parent": None}
        
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return {"error": "Path does not exist or is not a directory"}
            
        directories = []
        # List subdirectories
        for item in p.iterdir():
            if item.is_dir():
                directories.append(item.name)
        
        return {
            "current": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "directories": sorted(directories)
        }
    except Exception as e:
        return {"error": str(e)}

class ProcessRequest(BaseModel):
    video_path: str
    threshold: float = 50000.0

class RenderRequest(BaseModel):
    output_name: str
    crf: int = 23 # Default quality (lower is better, 18-28 is normal range)
    preset: str = "medium"
    save_to_source: bool = True # New flag

@app.post("/api/render")
async def render_video(req: RenderRequest):
    if process_state.is_running:
        raise HTTPException(status_code=400, detail="Processing in progress")
    
    segments_path = OUTPUT_DIR / "segments.json"
    if not segments_path.exists():
        raise HTTPException(status_code=404, detail="No extracted segments found. Run extraction first.")
        
    audio_temp_path = TEMP_DIR / "temp_audio.m4a"
    concat_list_path = OUTPUT_DIR / "concat_list.txt"

    try:
        process_state.is_running = True
        
        # 1. Load Metadata
        with open(segments_path, 'r') as f:
            data = json.load(f)
            # Handle both old format (list) and new format (dict) for backward compat slightly
            if isinstance(data, list):
                segments = data
                source_video = None
            else:
                segments = data["segments"]
                source_video = data.get("source_video")
            
        # 2. Extract Audio (if source known)
        has_audio = False
        if source_video and Path(source_video).exists():
            if audio_temp_path.exists(): audio_temp_path.unlink()
            
            # Extract audio: ffmpeg -i source -vn -c:a aac temp_audio.m4a
            # Using aac for broad compatibility
            mix_cmd = [
                FFMPEG_BIN, "-y", "-i", source_video, "-vn", "-c:a", "aac", str(audio_temp_path)
            ]
            proc = await asyncio.create_subprocess_exec(
                *mix_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            if proc.returncode == 0 and audio_temp_path.exists():
                has_audio = True

        # 3. Create Concat File
        # Fix for "First frame everywhere": Ensure paths are absolute and properly escaped
        # Also, for the LAST frame, we must repeat it effectively or give it duration.
        # The 'duration' directive applies to the file PRECEDING it.
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(segments):
                # Absolute path prevents CWD confusion
                abs_path = (OUTPUT_DIR / s['file']).resolve()
                # Windows path escaping for ffmpeg concat
                safe_path = str(abs_path).replace('\\', '/') 
                
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration {s['duration']}\n")
            
            # Important: FFmpeg concat sometimes drops the last frame if no duration follows?
            # Actually, standard practice for images is:
            # file 'A' \n duration 5 \n file 'B' \n duration 5 ...
            # AND THEN repeat the last file one more time without duration to serve as a closure?
            # Or just ensure the loop covers all.
            # Reference: https://trac.ffmpeg.org/wiki/Slideshow
            # "Due to a quirk, the last image has to be specified twice - the 2nd time without duration"
            if segments:
                last = segments[-1]
                abs_path = (OUTPUT_DIR / last['file']).resolve()
                safe_path = str(abs_path).replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        # 4. Output Path
        final_output_dir = RENDER_DIR

        
        logger.info(f"Render Request: save_to_source={req.save_to_source}, source_video={source_video}")
        
        if req.save_to_source and source_video:
             # Logic: Source Dir / Render Output
             src_path = Path(source_video)
             
             # Debug log
             logger.info(f"Checking source path: {src_path}, Exists: {src_path.exists()}")
             
             # If exact file exists OR parent dir exists (if file was moved but dir remains)
             if src_path.exists() or src_path.parent.exists():
                 custom_out = src_path.parent / "Render Output"
                 try:
                     custom_out.mkdir(parents=True, exist_ok=True)
                     final_output_dir = custom_out
                     logger.info(f"Set output directory to source subfolder: {final_output_dir}")
                 except Exception as e:
                     logger.error(f"Failed to create source output dir: {e}")
             else:
                 logger.warning(f"Source path invalid, falling back to default.")
        
        # Ensure dir exists
        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Name resolution
        final_name = req.output_name.strip()
        if not final_name:
             if source_video:
                  final_name = Path(source_video).name
             else:
                  final_name = "output.mp4"
        
        if not final_name.lower().endswith('.mp4'):
            final_name += ".mp4"
            
        output_file = final_output_dir / final_name
        
        # Log the decision
        logger.info(f"Rendering to: {output_file}")
        
        if output_file.exists():
            try:
                output_file.unlink()
                logger.info(f"Existing file removed: {output_file}")
            except Exception as e:
                logger.error(f"Failed to remove existing file: {e}")
            
        # 5. Construct Render Command
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path)
        ]
        
        input_map = ["-map", "0:v"]
        
        if has_audio:
            cmd.extend(["-i", str(audio_temp_path)])
            input_map.extend(["-map", "1:a"])
            # shortening audio to video length? or video to audio? 
            # -shortest might cut video if audio is shorter (likely due to silence removal? no).
            # We usually want video length to dictate.
        
        cmd.extend(input_map)
        
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(req.crf),
            "-preset", req.preset,
            "-vsync", "vfr",
            str(output_file)
        ])
        
        msg = f"Rendering to {output_file.name}..."
        logger.info(msg)
        await manager.broadcast_status({"type": "info", "message": msg})

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg Error: {stderr.decode()}")
            raise Exception(f"FFmpeg failed: {stderr.decode()[-300:]}")
            
        return {"status": "completed", "path": str(output_file)}

    except Exception as e:
        logger.error(f"Render error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        process_state.is_running = False
        # Cleanup
        if 'concat_list_path' in locals() and concat_list_path.exists():
            try: concat_list_path.unlink()
            except: pass
        if 'audio_temp_path' in locals() and audio_temp_path.exists():
            try: audio_temp_path.unlink()
            except: pass

@app.post("/api/process/start")
async def start_process(req: ProcessRequest):
    if process_state.is_running:
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    # Run in background
    asyncio.create_task(process_video_task(req.video_path, req.threshold))
    return {"status": "started"}

@app.post("/api/process/stop")
def stop_process():
    if process_state.is_running:
        process_state.should_stop = True
        return {"status": "stopping"}
    return {"status": "not_running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Installer Routes ---
@app.get("/api/setup/check")
def check_setup():
    ffmpeg_path = resolve_binary("ffmpeg")
    probe_path = resolve_binary("ffprobe")
    
    # Simple check if they resolve to something existing
    has_ffmpeg = ffmpeg_path and Path(ffmpeg_path).exists()
    
    return {
        "installed": has_ffmpeg,
        "ffmpeg_path": ffmpeg_path,
        "platform": installer.get_platform()
    }

@app.post("/api/setup/install")
async def run_install():
    if process_state.install_status in ["downloading", "extracting"]:
        return {"status": "busy"}

    def progress_handler(pct, msg):
        process_state.install_progress = pct
        process_state.install_msg = msg
        
    def install_thread():
        try:
            process_state.install_status = "downloading"
            process_state.install_msg = "Starting..."
            bin_path = installer.install_ffmpeg(FFMPEG_DIR, progress_callback=progress_handler)
            
            # Update global refs
            global FFMPEG_BIN, FFPROBE_BIN
            FFMPEG_BIN = resolve_binary("ffmpeg")
            FFPROBE_BIN = resolve_binary("ffprobe")
            
            process_state.install_status = "complete"
            process_state.install_progress = 100
            process_state.install_msg = f"Installed at {bin_path}"
        except Exception as e:
            process_state.install_status = "error"
            process_state.install_msg = str(e)

    # Convert to async task
    asyncio.get_event_loop().run_in_executor(None, install_thread)
    return {"status": "started"}

@app.get("/api/setup/status")
def get_install_status():
    return {
        "status": process_state.install_status,
        "progress": process_state.install_progress,
        "message": process_state.install_msg
    }

# --- Transformations (Logo Removal) ---
class LogoRemovalRequest(BaseModel):
    x: int
    y: int
    width: int = 150
    height: int = 50
    method: str = "inpaint"
    selected_files: List[str] = [] # Optional: if empty, process all? Or strict?
def process_logo_removal(img_path, x, y, w, h):
    try:
        img = cv2.imread(str(img_path))
        if img is None: 
            logger.error(f"Failed to load image: {img_path}")
            return False
        
        # Smart Masking using Reference Logo
        ref_path = BASE_DIR / "logo_reference.png"
        mask = None
                
        if ref_path.exists():
            try:
                # Load reference
                ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
                if ref_img is not None:
                    # Option A: Threshold (Invert for dark text)
                    _, logo_mask = cv2.threshold(ref_img, 200, 255, cv2.THRESH_BINARY_INV)
                    
                    # Check if we got anything
                    if cv2.countNonZero(logo_mask) < 10:
                        # Maybe it's light text? Try opposite
                        _, logo_mask = cv2.threshold(ref_img, 50, 255, cv2.THRESH_BINARY)
                    
                    if cv2.countNonZero(logo_mask) > 10:
                        # Dilate to cover edges
                        kernel = np.ones((5,5), np.uint8)
                        logo_mask = cv2.dilate(logo_mask, kernel, iterations=2)
                        
                        # Create full size mask
                        mask = np.zeros(img.shape[:2], np.uint8)
                        
                        # Place logo mask at coordinates
                        h_logo, w_logo = logo_mask.shape
                        h_img, w_img = img.shape[:2]
                        
                        x_end = min(x + w_logo, w_img)
                        y_end = min(y + h_logo, h_img)
                        
                        w_use = x_end - x
                        h_use = y_end - y
                        
                        if w_use > 0 and h_use > 0:
                            mask[y:y_end, x:x_end] = logo_mask[0:h_use, 0:w_use]
                            logger.info(f"Smart Mask Applied: {x},{y}")
            except Exception as e:
                logger.error(f"Smart mask error: {e}")

        # Fallback OR Validation
        if mask is None or cv2.countNonZero(mask) < 10:
            logger.warning(f"Smart mask failed or too small. FALLBACK to Rectangle at {x},{y} {w}x{h}")
            mask = np.zeros(img.shape[:2], np.uint8)
            h_img, w_img = img.shape[:2]
            x2 = min(x + w, w_img)
            y2 = min(y + h, h_img)
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            mask[y:y2, x:x2] = 255
        
        # Double check mask validity
        if cv2.countNonZero(mask) == 0:
                logger.error("Mask is still empty! Coordinates might be out of bounds.")
                return False

        # Inpaint
        result = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
        cv2.imwrite(str(img_path), result)
        return True
    except Exception as e:
        logger.error(f"Inpaint critical error: {e}")
        return False

@app.post("/api/process/remove_logo")
async def remove_logo(req: LogoRemovalRequest):
    if process_state.is_running:
        return {"error": "Busy"}
        
    try:
        process_state.is_running = True
        
        # Get all frames first
        all_frames = sorted(list(OUTPUT_DIR.glob("frame_*.png")))
        
        # Filter if selection is provided
        if req.selected_files:
            target_set = set(req.selected_files)
            frames = [f for f in all_frames if f.name in target_set]
        else:
            # STRICT MODE: If no selection provided, do NOTHING.
            # User explicitly requested to avoid touching unselected images.
            return {"status": "skipped", "message": "No frames selected"}
        
        if not frames:
             return {"status": "completed", "processed": 0, "message": "No frames selected"}

        count = 0
        total = len(frames)
        
        for i, f in enumerate(frames):
            if process_logo_removal(f, req.x, req.y, req.width, req.height):
                count += 1
            
            # Progress update every 10 frames
            if i % 10 == 0:
                pct = int((i / total) * 100)
                await manager.broadcast_status({
                    "type": "progress", 
                    "progress": pct, 
                    "current_time": f"Logo: {i}/{total}"
                })
                # Check for stop signal? (Not implemented for this loop deeply yet)

        await manager.broadcast_status({"type": "complete", "subtype": "logo_removal", "count": count})
        return {"status": "completed", "processed": count}
        
    except Exception as e:
        logger.error(f"Logo removal error: {e}")
        return {"error": str(e)}
    finally:
        process_state.is_running = False

@app.on_event("startup")
async def startup_event():
    # Check for FFmpeg on startup
    if not resolve_binary("ffmpeg").endswith("ffmpeg") and not resolve_binary("ffmpeg").endswith(".exe"):
         # If resolve returns just the name, it might be in path, let's explicit check
         if not shutil.which("ffmpeg"):
             logger.warning("FFmpeg not found! Please use the Web UI to install it.")
    else:
        logger.info(f"FFmpeg detected: {resolve_binary('ffmpeg')}")

@app.get("/api/project/load")
def load_existing_project():
    """Loads existing frames from segments.json if available."""
    seg_path = OUTPUT_DIR / "segments.json"
    if not seg_path.exists():
        return {"exists": False}
        
    try:
        with open(seg_path, 'r') as f:
            data = json.load(f)
            
        # Handle format variations
        if isinstance(data, list):
            segments = data
            source = ""
        else:
            segments = data.get("segments", [])
            source = data.get("source_video", "")
            
        # Validate that mostly images exist (quick check)
        if segments and not (OUTPUT_DIR / segments[0]['file']).exists():
             return {"exists": False, "reason": "Files missing"}

        # Transform to frontend format if needed
        # Frontend expects: { filename, url, duration, start, end }
        # stored: { file, duration }
        # We need to reconstruct start/end times roughly
        
        frontend_frames = []
        current_time = 0.0
        
        for s in segments:
            dur = s['duration']
            end_time = current_time + dur
            
            frontend_frames.append({
                "filename": s['file'],
                "url": f"/frames/{s['file']}",
                "start": format_time(current_time),
                "end": format_time(end_time),
                "duration": f"{dur:.2f}s"
            })
            current_time = end_time
            
        return {
            "exists": True,
            "source": source,
            "count": len(frontend_frames),
            "frames": frontend_frames
        }
            
    except Exception as e:
        logger.error(f"Project load error: {e}")
        return {"exists": False, "error": str(e)}

@app.post("/api/project/clear")
def clear_project_files():
    """Deletes all extracted frames and metadata."""
    if process_state.is_running:
         raise HTTPException(status_code=400, detail="Cannot clear while processing.")
         
    count = 0
    try:
        # Delete frames
        for f in OUTPUT_DIR.glob("frame_*.png"):
            try:
                f.unlink()
                count += 1
            except: pass
            
        # Delete metadata
        for f in ["segments.json", "metadata.json", "concat_list.txt"]:
            p = OUTPUT_DIR / f
            if p.exists(): p.unlink()
            
            # Check root metadata too
            p_root = BASE_DIR / "metadata.json"
            if p_root.exists(): p_root.unlink()
            
        return {"status": "cleared", "deleted_count": count}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 to allow external access if needed, but 127.0.0.1 is safer for local
    uvicorn.run(app, host="127.0.0.1", port=8000)
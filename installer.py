import os
import sys
import platform
import shutil
import logging
import urllib.request
import zipfile
import tarfile
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("installer")

# URLs for static builds (Pinned versions for stability)
# Windows: Gyan.dev (git-master release usually stable enough, or use release-essentials)
WIN_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
# Linux: John Van Sickle (amd64 is most common)
LINUX_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"

def get_platform():
    return platform.system().lower()

def is_ffmpeg_installed(custom_path=None):
    """Checks if ffmpeg is available in PATH or custom_path."""
    if custom_path:
        bin_name = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
        local_bin = custom_path / bin_name
        if local_bin.exists():
            return str(local_bin)
        
        # Check subdirectories (common in extracted zips)
        # e.g. ffmpeg-5.1-essentials_build/bin/ffmpeg.exe
        found = list(custom_path.rglob(bin_name))
        if found:
            return str(found[0])

    return shutil.which("ffmpeg")

def install_ffmpeg(target_dir: Path, progress_callback=None):
    """
    Downloads and extracts FFmpeg to target_dir.
    progress_callback(percent, status_message)
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    os_type = get_platform()
    
    try:
        if os_type == "windows":
            url = WIN_URL
            filename = "ffmpeg.zip"
        elif os_type == "linux":
            url = LINUX_URL
            filename = "ffmpeg.tar.xz"
        else:
            raise Exception(f"Unsupported OS: {os_type}")

        filepath = target_dir / filename
        
        # 1. Download
        if progress_callback: progress_callback(0, "Downloading FFmpeg...")
        
        def _report(block_num, block_size, total_size):
            if total_size > 0:
                percent = int((block_num * block_size * 100) / total_size)
                if progress_callback: progress_callback(percent, f"Downloading... {percent}%")

        logger.info(f"Downloading from {url} to {filepath}")
        urllib.request.urlretrieve(url, filepath, reporthook=_report)
        
        # 2. Extract
        if progress_callback: progress_callback(100, "Extracting...")
        logger.info("Extracting...")
        
        if filename.endswith(".zip"):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif filename.endswith(".tar.xz"):
            with tarfile.open(filepath, "r:xz") as tar:
                tar.extractall(target_dir)
                
        # 3. Cleanup
        os.remove(filepath)
        
        # 4. Find binary and set permissions (Linux)
        bin_path = is_ffmpeg_installed(target_dir)
        if not bin_path:
            raise Exception("Extraction finished but binary not found.")
            
        if os_type == "linux":
            # Ensure executable
            st = os.stat(bin_path)
            os.chmod(bin_path, st.st_mode | 0o111)
            
        if progress_callback: progress_callback(100, "Installation Complete")
        return bin_path

    except Exception as e:
        logger.error(f"Installation failed: {e}")
        raise e

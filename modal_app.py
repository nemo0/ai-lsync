import modal
import os
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import requests
from urllib.parse import urlparse
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Response, HTTPException

# --- Modal Setup ---
app = modal.App("latentsync-api")
volume = modal.NetworkFileSystem.from_name("latentsync-checkpoints-vol", create_if_missing=True)

# --- Paths ---
REMOTE_CODE_PATH = Path("/app")
CHECKPOINT_PATH = Path("/checkpoints")

# --- Download Required Models ---
def _download_models():
    from huggingface_hub import hf_hub_download

    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_PATH / "whisper").mkdir(exist_ok=True)

    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="latentsync_unet.pt",
        local_dir=CHECKPOINT_PATH,
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="whisper/tiny.pt", # Correct path within the repo
        local_dir=CHECKPOINT_PATH,
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="stable_syncnet.pt",
        local_dir=CHECKPOINT_PATH,
        local_dir_use_symlinks=False,
    )
    print("✅ All models downloaded.")

# --- Modal Image Configuration ---
latentsync_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .run_commands("pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121")
    .pip_install(
        "diffusers==0.32.2",
        "transformers==4.48.0",
        "decord==0.6.0",
        "accelerate==0.26.1",
        "einops==0.7.0",
        "omegaconf==2.3.0",
        "opencv-python==4.9.0.80",
        "mediapipe==0.10.11",
        "python_speech_features==0.6",
        "librosa==0.10.1",
        "scenedetect[opencv]==0.6.1",
        "ffmpeg-python==0.2.0",
        "imageio==2.31.1",
        "imageio-ffmpeg==0.5.1",
        "lpips==0.1.4",
        "face-alignment==1.4.1",
        "gradio==5.24.0",
        "huggingface-hub==0.30.2",
        "numpy==1.26.4",
        "kornia==0.8.0",
        "insightface==0.7.3",
        "onnxruntime-gpu==1.21.0",
        "DeepCache==0.1.1",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "requests"
    )
    .run_function(
        _download_models,
        network_file_systems={str(CHECKPOINT_PATH): volume},
    )
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(".", remote_path="/app")
    .add_local_dir("scripts", remote_path="/app/scripts")
    .add_local_dir("configs", remote_path="/app/configs")
)

# --- GPU Class for Inference ---
@app.cls(
    gpu="A10G",
    image=latentsync_image,
    network_file_systems={str(CHECKPOINT_PATH): volume},
    timeout=1800,
)
class LatentSync:
    @modal.enter()
    def setup(self):
        import sys
        from omegaconf import OmegaConf

        sys.path.append("/app")  # Ensure scripts import works
        os.chdir(REMOTE_CODE_PATH)

        config_path = REMOTE_CODE_PATH / "configs/unet/stage2_512.yaml"
        self.config = OmegaConf.load(config_path)

        print("✅ Setup complete. Current working directory:", os.getcwd())
        print("📂 Scripts directory:", os.listdir("/app/scripts"))

    @modal.method()
    def generate(
        self,
        video_bytes: bytes,
        audio_bytes: bytes,
        video_filename: str,
        audio_filename: str,
        guidance_scale: float,
        inference_steps: int,
        seed: int,
    ) -> bytes:
        from scripts.inference import main as inference_main

        temp_input_dir = Path("/tmp/input")
        temp_input_dir.mkdir(parents=True, exist_ok=True)

        video_path = temp_input_dir / video_filename
        audio_path = temp_input_dir / audio_filename

        with open(video_path, "wb") as f:
            f.write(video_bytes)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        output_dir = Path("/tmp/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = output_dir / f"result_{current_time}.mp4"

        args = argparse.Namespace(
            inference_ckpt_path=str(CHECKPOINT_PATH / "latentsync_unet.pt"),
            video_path=str(video_path),
            audio_path=str(audio_path),
            video_out_path=str(output_video_path),
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            temp_dir=str(output_dir),
            enable_deepcache=True,
        )

        run_config = deepcopy(self.config)
        run_config["run"].update({
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        })

        try:
            inference_main(config=run_config, args=args)
            with open(output_video_path, "rb") as f:
                return f.read()
        except Exception as e:
            print("❌ Inference failed:", e)
            raise

# --- FastAPI Web Server ---
fastapi_app = FastAPI()

def _get_filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)

@fastapi_app.post("/lipsync", response_class=Response)
async def lipsync(
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
    guidance_scale: float = Form(1.5),
    inference_steps: int = Form(20),
    seed: int = Form(1247),
):
    video_bytes = audio_bytes = None
    video_filename = audio_filename = None

    if video and video_url:
        raise HTTPException(status_code=400, detail="Only one of video or video_url allowed.")
    if video:
        video_bytes = await video.read()
        video_filename = video.filename
    elif video_url:
        r = requests.get(video_url)
        r.raise_for_status()
        video_bytes = r.content
        video_filename = _get_filename_from_url(video_url)

    if audio and audio_url:
        raise HTTPException(status_code=400, detail="Only one of audio or audio_url allowed.")
    if audio:
        audio_bytes = await audio.read()
        audio_filename = audio.filename
    elif audio_url:
        r = requests.get(audio_url)
        r.raise_for_status()
        audio_bytes = r.content
        audio_filename = _get_filename_from_url(audio_url)

    if not video_bytes or not audio_bytes:
        raise HTTPException(status_code=400, detail="Video and audio inputs are required.")

    model = LatentSync()
    output = model.generate.remote(
        video_bytes,
        audio_bytes,
        video_filename,
        audio_filename,
        guidance_scale,
        inference_steps,
        seed
    )
    return Response(content=output, media_type="video/mp4")

# --- ASGI App Entry Point ---
@app.function(
    image=latentsync_image,
    network_file_systems={str(CHECKPOINT_PATH): volume},
    timeout=900,
)
@modal.asgi_app()
def web_server():
    return fastapi_app
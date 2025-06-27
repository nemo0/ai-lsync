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

# --- App and Volume Setup ---
app = modal.App("latentsync-api")
# This is a persistent shared volume to store model checkpoints
volume = modal.NetworkFileSystem.from_name("latentsync-checkpoints-vol", create_if_missing=True)

# --- Paths ---
# We define remote paths for the code and checkpoints
REMOTE_CODE_PATH = Path("/app")
CHECKPOINT_PATH = Path("/checkpoints")

# --- Model Download Logic ---
# This function runs once during the image build to download and cache the models
def _download_models():
    from huggingface_hub import hf_hub_download

    # Ensure the target directory exists
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_PATH / "whisper").mkdir(exist_ok=True)

    # Download the main UNet model
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="latentsync_unet.pt",
        local_dir=CHECKPOINT_PATH,
        local_dir_use_symlinks=False,
    )
    # Download the Whisper model
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="whisper/tiny.pt",
        local_dir=CHECKPOINT_PATH,
        local_dir_use_symlinks=False,
    )
    # Download the SyncNet model
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="stable_syncnet.pt",
        local_dir=CHECKPOINT_PATH,
        local_dir_use_symlinks=False,
    )
    print("All models downloaded successfully.")

# --- Modal Image Definition ---
# This defines the container environment, installing all necessary dependencies
latentsync_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg") # ffmpeg is essential for video processing
    # First, install PyTorch with the specific CUDA version via a direct command
    .run_commands("pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121")
    # Then, install the rest of the packages
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
        "requests", # Added for downloading files from URLs
    )
    # Run the model download function after dependencies are installed
    .run_function(
        _download_models,
        network_file_systems={str(CHECKPOINT_PATH): volume}, # Mount the volume and run the download
    )
    # Set the PYTHONPATH to include the app directory, making local modules importable
    .env({"PYTHONPATH": str(REMOTE_CODE_PATH)})
    # Add local code last to optimize build speed on file changes
    .add_local_dir(".", remote_path=str(REMOTE_CODE_PATH))
)

# --- Inference Class ---
# This class encapsulates the model and the inference logic.
# It's decorated with @app.cls to run on a GPU-equipped container on Modal.
@app.cls(
    gpu="A10G", # A10G has 24GB VRAM, suitable for the 18GB requirement
    image=latentsync_image,
    network_file_systems={str(CHECKPOINT_PATH): volume},
    timeout=600, # Set a 10-minute timeout for inference
)
class LatentSync:
    @modal.enter()
    def setup(self):
        """
        This method runs once when the container for the class starts.
        We change the directory and load the model configuration.
        """
        from omegaconf import OmegaConf
        
        os.chdir(REMOTE_CODE_PATH)
        print(f"Current working directory: {os.getcwd()}")
        
        # Load the configuration file for the model
        config_path = REMOTE_CODE_PATH / "configs" / "unet" / "stage2_512.yaml"
        self.config = OmegaConf.load(config_path)
        print("Model configuration loaded.")

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
        """
        The main inference method. It takes paths to video/audio and generation
        parameters, runs the lipsync pipeline, and returns the output video as bytes.
        """
        from scripts.inference import main as inference_main

        # Create temporary directories for input and output
        temp_input_dir = Path("/tmp/input")
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_input_dir / video_filename
        audio_path = temp_input_dir / audio_filename

        # Write the received bytes to temporary files in this container
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        output_dir = Path("/tmp/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = str(output_dir / f"result_{current_time}.mp4")

        # Create an arguments namespace object, using the new temp file paths
        args = argparse.Namespace(
            inference_ckpt_path=str(CHECKPOINT_PATH / "latentsync_unet.pt"),
            video_path=str(video_path),
            audio_path=str(audio_path),
            video_out_path=output_video_path,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            temp_dir=str(output_dir),
            enable_deepcache=True, # Enable for better performance
        )
        
        # It's good practice to work with a copy of the config for each run
        # to avoid stateful issues if a container is reused.
        run_config = deepcopy(self.config)
        
        # Update the config copy with runtime arguments
        run_config["run"].update({
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        })
        
        print("Running inference...")
        try:
            # Call the main inference function from the original script
            inference_main(config=run_config, args=args)
            print("Inference complete.")
            
            # Read the generated video file and return its content
            with open(output_video_path, "rb") as f:
                content = f.read()
            return content
            
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

# --- FastAPI Web Server ---
# We define a FastAPI app to handle web requests.
fastapi_app = FastAPI()

def _get_filename_from_url(url: str) -> str:
    """Helper function to extract a filename from a URL."""
    return os.path.basename(urlparse(url).path)

@fastapi_app.post("/lipsync", response_class=Response)
async def lipsync(
    # --- Input Options ---
    # User can provide either a file upload OR a URL for video and audio
    video: Optional[UploadFile] = File(None, description="Video file to be lip-synced."),
    audio: Optional[UploadFile] = File(None, description="Audio file to sync with the video."),
    video_url: Optional[str] = Form(None, description="URL of the video file."),
    audio_url: Optional[str] = Form(None, description="URL of the audio file."),
    
    # --- Generation Parameters ---
    guidance_scale: float = Form(1.5, description="Classifier-free guidance scale."),
    inference_steps: int = Form(20, description="Number of DDIM inference steps."),
    seed: int = Form(1247, description="Random seed for generation."),
):
    """
    This endpoint performs lip-syncing on a video using a target audio.
    It accepts either direct file uploads or URLs for the video and audio sources.
    """
    video_bytes: Optional[bytes] = None
    video_filename: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    audio_filename: Optional[str] = None

    # --- Step 1: Process Video Input (File or URL) ---
    if video and video_url:
        raise HTTPException(status_code=400, detail="Provide either a video file or a video_url, not both.")
    
    if video:
        video_bytes = await video.read()
        video_filename = video.filename
    elif video_url:
        try:
            print(f"Downloading video from: {video_url}")
            response = requests.get(video_url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            video_bytes = response.content
            video_filename = _get_filename_from_url(video_url) or "video_from_url.mp4"
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video from URL: {e}")
    else:
        raise HTTPException(status_code=400, detail="Either a video file or video_url must be provided.")

    # --- Step 2: Process Audio Input (File or URL) ---
    if audio and audio_url:
        raise HTTPException(status_code=400, detail="Provide either an audio file or an audio_url, not both.")

    if audio:
        audio_bytes = await audio.read()
        audio_filename = audio.filename
    elif audio_url:
        try:
            print(f"Downloading audio from: {audio_url}")
            response = requests.get(audio_url)
            response.raise_for_status()
            audio_bytes = response.content
            audio_filename = _get_filename_from_url(audio_url) or "audio_from_url.mp3"
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {e}")
    else:
        raise HTTPException(status_code=400, detail="Either an audio file or audio_url must be provided.")
    
    # --- Step 3: Run Inference ---
    # Instantiate the Modal class and call the generation method remotely
    model = LatentSync()
    output_video_bytes = model.generate.remote(
        video_bytes,
        audio_bytes,
        video_filename,
        audio_filename,
        guidance_scale,
        inference_steps,
        seed
    )
    
    # --- Step 4: Return Result ---
    # Return the generated video as a response
    return Response(content=output_video_bytes, media_type="video/mp4")

# --- Modal ASGI App ---
# This serves the FastAPI application using Modal's web hosting capabilities.
@app.function(
    image=latentsync_image,
    network_file_systems={str(CHECKPOINT_PATH): volume},
    timeout=900,
)
@modal.asgi_app()
def web_server():
    return fastapi_app

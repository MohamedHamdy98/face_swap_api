from flask import Flask, request, jsonify
import os
import subprocess
import gdown
from tqdm import tqdm

app = Flask(__name__)

ROOP_DIR = "/tmp/roop"  # Use /tmp for Netlify Functions
MODEL_PATH = os.path.join(ROOP_DIR, "models/inswapper_128.onnx")

def download_from_google_drive(url, output_path):
    try:
        file_id = url.split("/d/")[1].split("/view")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Failed to download file from {url}. Error: {str(e)}")

def is_package_installed(package_name):
    try:
        subprocess.run(["pip", "show", package_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def setup_roop():
    os.makedirs(ROOP_DIR, exist_ok=True)
    os.chdir(ROOP_DIR)

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading inswapper_128.onnx...")
        with tqdm(total=100, desc="Downloading model") as pbar:
            subprocess.run(["wget", "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx", "-O", "inswapper_128.onnx"])
            pbar.update(100)
        os.makedirs("models", exist_ok=True)
        subprocess.run(["mv", "inswapper_128.onnx", "./models/"])
    else:
        print("Model is already downloaded.")

    if is_package_installed("onnxruntime-gpu"):
        print("onnxruntime-gpu is already installed.")
    else:
        print("Installing onnxruntime-gpu...")
        with tqdm(total=100, desc="Installing onnxruntime-gpu") as pbar:
            subprocess.run(["pip", "install", "onnxruntime-gpu"])
            pbar.update(100)

    if is_package_installed("torch") and is_package_installed("torchvision") and is_package_installed("torchaudio"):
        print("Torch packages are already installed.")
    else:
        print("Installing torch, torchvision, and torchaudio...")
        with tqdm(total=100, desc="Installing PyTorch") as pbar:
            subprocess.run(["pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y"])
            subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "--force-reinstall", "--index-url", "https://download.pytorch.org/whl/cu118"])
            pbar.update(100)

setup_roop()

@app.route('/api/swap', methods=['POST'])
def face_swap():
    try:
        os.chdir(ROOP_DIR)

        target_url = request.form.get('target_url')
        source_url = request.form.get('source_url')
        output_path = '/tmp/uploaded_data/outputs/output_face_swap.mp4'

        target_path = "/tmp/uploaded_data/videos/target_video.mp4"
        source_path = "/tmp/uploaded_data/images/source_image.jpg"

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.makedirs(os.path.dirname(source_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("Downloading target video...")
        download_from_google_drive(target_url, target_path)
        print("Downloading source image...")
        download_from_google_drive(source_url, source_path)

        print("Performing face swapping...")
        with tqdm(total=100, desc="Face swapping") as pbar:
            command = f"python run.py --target {target_path} --source {source_path} -o {output_path} --execution-provider cuda --frame-processor face_swapper"
            subprocess.run(command, shell=True, check=True)
            pbar.update(100)

        return jsonify({
            'status': 'success',
            'message': 'Face swapping completed',
            'output_path': output_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Entry point for Netlify Functions
def handler(event, context):
    with app.app_context():
        return app.full_dispatch_request()

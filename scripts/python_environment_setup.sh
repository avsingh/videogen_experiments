# Create virtual environment
python3 -m venv wan_env
source wan_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch for CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install WAN 2.2 requirements
pip install -r requirements.txt

# Install additional tools for fine-tuning
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install transformers==4.36.0
pip install diffusers==0.25.0
pip install xformers  # For memory-efficient attention
pip install wandb
pip install tensorboard

# Data processing
pip install opencv-python
pip install pillow
pip install imageio
pip install imageio-ffmpeg

# Google Photos API
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

# Verify installation
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF

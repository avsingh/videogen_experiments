# Clone the repository
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# Install dependencies
pip install -r requirements.txt

# Download the TI2V-5B model
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."

#############
# Compatibility Issues
### ATTEMPT 1
# First check your PyTorch version
python -c "import torch; print(torch.__version__)"

# Reinstall flash_attn to match your PyTorch/CUDA setup
pip uninstall flash_attn -y
pip install flash-attn --no-build-isolation

### ATTEMPT 2
# Uninstall current PyTorch and flash_attn
pip uninstall torch torchvision torchaudio flash_attn -y

# Install PyTorch 2.4.0 (which the requirements recommend)
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Now install flash_attn
pip install flash-attn --no-build-isolation

### ATTEMPT 3
# Check versions
pip list | grep -E "torch|flash"

# Complete clean reinstall
pip uninstall torch torchvision torchaudio flash-attn flash_attn -y
pip cache purge

# Install PyTorch 2.4.1 with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn (this will take several minutes to compile)
pip install flash-attn==2.6.3 --no-build-isolation

### ATTEMPT 4
pip install decord

### ATTEMPT 5
pip install librosa

### ATTEMPT 6
pip install peft av soundfile imageio-ffmpeg

### ATTEMPT 7
pip install --upgrade transformers

### ATTEMPT 8
pip install transformers==4.44.2

### ATTEMPT 9
pip install peft==0.11.1

### ATTEMPT 10
pip install diffusers==0.30.3 peft==0.13.2

#############
scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ubuntu@209.20.156.222:~/Wan2.2/'ti2v-5B_1280*704_1_Summer_beach_vacation_style,_a_white_cat_wearing_s_20251127_210926.mp4' ~/src/wan-fine-tune/generated/

scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem /Users/av/src/wan-fine-tune/manual_downloaded_google_photos/IMG_2041.JPG ubuntu@209.20.156.222:~/Wan2.2

python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --image /home/ubuntu/Wan2.2/IMG_2041.JPG \
  --prompt "Make the LED man in the foreground dance with the background fireworks pulsating"


# Downloading the bigger model
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B

# Generate using the larger model
python generate.py \
  --task i2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --image your-image.jpg \
  --prompt "your prompt"

####### Building the app
# Now transfering the app
scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem /Users/av/src/wan-fine-tune/app/wan_web_app.tar.gz ubuntu@209.20.156.222:~/Wan2.2

cd ~/Wan2.2
tar -xzf wan_web_app.tar.gz
pip install flask
mkdir -p uploads

# Attempt 1
Not accessible from the public internet (http://209.20.156.222:5000/)

# Attempt 2
ssh -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem -L 5000:localhost:5000 ubuntu@209.20.156.222

####### Fine tuning
cd ~
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Create models directory
mkdir -p models/wan2.2

# Download the models (this will take a while - ~100GB total)
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir models/wan2.2
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir models/wan2.2
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir models/wan2.2
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/text_encoders/umt5_xxl_fp16.safetensors --local-dir models/wan2.2
```

This download will take **1-2 hours** and use ~100GB of disk space.

## Step 5: Meanwhile, Let's Prepare Your Dataset

While the models download, tell me:

1. **What style do you want to achieve?**
2. **Do you have training data, or should we find some together?**
3. **Images or videos?** (Images are easier for pure visual style)

Once you decide, I'll help you:
- Organize the dataset structure
- Write captions for each sample
- Configure the training parameters

## Quick Example Dataset Structure:
```
ai-toolkit/
└── datasets/
    └── my_style/
        ├── sample_001.jpg
        ├── sample_001.txt (caption)
        ├── sample_002.jpg
        ├── sample_002.txt
        └── ... (20-30 total)


###### Fine-tuning with ai-toolkit (FAILURE)
scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem /Users/av/src/wan-fine-tune/sunset_training_data/* ubuntu@209.20.156.222:~/sunset_training_data

cd ~/ai-toolkit
mkdir -p datasets/sunset_style
# Move your uploaded images
mv ~/sunset_training_data/* datasets/sunset_style/

scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ~/src/wan-fine-tune/lambda/create_captions.py ubuntu@209.20.156.222:~/sunset_training_data

python ~/sunset_training_data/create_captions.py
	
scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ~/src/wan-fine-tune/lambda/sunset_lora_high.yaml ubuntu@209.20.156.222:~/ai-toolkit/config/sunset_lora_high.yaml

scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ~/src/wan-fine-tune/lambda/sunset_lora_low.yaml ubuntu@209.20.156.222:~/ai-toolkit/config/sunset_lora_low.yaml

cd ~/ai-toolkit
source venv/bin/activate
python run.py config/sunset_lora_high.yaml

### ATTEMPT 1
pip install torchaudio

### ATTEMPT 2
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

### ATTEMPT 3
scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ~/src/wan-fine-tune/lambda/sunset_lora_high.yaml ubuntu@209.20.156.222:~/ai-toolkit/config/sunset_lora_high.yaml

###### Fine-tuning with diffusion-pipe (FAILED)
cd ~
git clone https://github.com/huggingface/diffusion-pipe.git
cd diffusion-pipe
pip install -e .

###### Fine-tuning with ostris/ai-toolkit
cd ~
git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout feature-wan-2-2

python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install protobuf six

# Download text encoder
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth --local-dir models/text_encoders

# Download VAE
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir models/vae

# Download high-noise transformer
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir models/diffusion_models

# Download low-noise transformer
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir models/diffusion_models

mkdir -p dataset/sunset_style
cp ~/ai-toolkit/datasets/sunset_style/* dataset/sunset_style/

scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ~/src/wan-fine-tune/lambda/dataset.toml ubuntu@209.20.156.222:/home/ubuntu/musubi-tuner/dataset

cd ~/musubi-tuner
source venv/bin/activate

accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --vae models/vae/split_files/vae/wan_2.1_vae.safetensors \
  --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
  --dataset_config dataset/dataset.toml \
  --output_dir output/sunset_high \
  --output_name sunset_style_high \
  --save_model_as safetensors \
  --prior_loss_weight 1.0 \
  --max_train_steps 3000 \
  --learning_rate 3e-4 \
  --optimizer_type adamw8bit \
  --xformers \
  --mixed_precision fp16 \
  --fp8_base \
  --gradient_checkpointing \
  --save_every_n_steps 500 \
  --network_module networks.lora_wan \
  --network_dim 32 \
  --network_alpha 32 \
  --timestep_sampling shift \
  --discrete_flow_shift 1.0 \
  --max_data_loader_n_workers 2 \
  --seed 42

### ATTEMPT 1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

### ATTEMPT 2
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --vae models/vae/split_files/vae/wan_2.1_vae.safetensors \
  --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
  --dataset_config dataset/dataset.toml \
  --output_dir output/sunset_high \
  --output_name sunset_style_high \
  --max_train_steps 3000 \
  --learning_rate 3e-4 \
  --optimizer_type adamw8bit \
  --xformers \
  --mixed_precision fp16 \
  --fp8_base \
  --gradient_checkpointing \
  --save_every_n_steps 500 \
  --network_module networks.lora_wan \
  --network_dim 32 \
  --network_alpha 32 \
  --timestep_sampling shift \
  --discrete_flow_shift 1.0 \
  --max_data_loader_n_workers 2 \
  --seed 42

####### After giving up on fine tuning, trying video concatenation

ssh -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem ubuntu@209.20.156.222
sudo apt update
sudo apt install -y ffmpeg

chmod +x stitch_videos.py

# Example usage
python stitch_videos.py output_combined.mp4 ti2v-5B_*.mp4

# generating a chain of videos
scp -i /Users/av/src/wan-fine-tune/lambda/wan-finetune-key.pem ~/src/wan-fine-tune/lambda/chain_generate.py ubuntu@209.20.156.222:/home/ubuntu/Wan2.2

chmod +x chain_generate.py

# Generate a 3-segment chain (will take ~4-6 minutes total)
python chain_generate.py examples/i2v_input.JPG "beautiful sunset over the ocean, golden hour lighting, cinematic" 3

# setting up GIT
echo "# videogen_experiments" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/avsingh/videogen_experiments.git
git push -u origin main

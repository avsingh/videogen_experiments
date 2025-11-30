# Update system
sudo apt update && sudo apt upgrade -y

# Install utilities
sudo apt install -y git wget curl tmux htop nvtop tree

# Check CUDA
nvcc --version  # Should be CUDA 12.1 or similar
which python3   # /usr/bin/python3

# Create workspace
mkdir -p /home/ubuntu/wan-finetune
cd /home/ubuntu/wan-finetune

# Clone WAN 2.2
git clone https://github.com/Wan-Video/Wan2.2
cd Wan2.2

# Explore the repo structure
tree -L 2

# Check README for specific setup instructions
cat README.md

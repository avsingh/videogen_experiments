# SSH into Lambda
ssh ubuntu@<lambda-ip>

# Navigate to workspace
cd /home/ubuntu/wan-finetune

# Extract dataset
tar -xzf google_photos_processed.tar.gz

# Verify
ls -lh google_photos_processed/
cat google_photos_processed/metadata.json | head -20

# Check dataset size
du -sh google_photos_processed/

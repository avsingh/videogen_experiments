# Create tar archive (faster to upload)
tar -czf google_photos_processed.tar.gz google_photos_processed/

# Check size
du -sh google_photos_processed.tar.gz

# Upload to Lambda (replace with your IP)
rsync -avz --progress google_photos_processed.tar.gz ubuntu@<lambda-ip>:/home/ubuntu/wan-finetune/

# Alternative: use scp
# scp google_photos_processed.tar.gz ubuntu@<lambda-ip>:/home/ubuntu/wan-finetune/

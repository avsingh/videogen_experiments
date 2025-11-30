cd /home/ubuntu/wan-finetune/Wan2.2

cat > wan_dataset.py << 'EOF'
import torch
from torch.utils.data import Dataset
import cv2
import json
import numpy as np
from pathlib import Path

class GooglePhotosVideoDataset(Dataset):
    """
    Custom dataset for WAN 2.2 training with Google Photos videos
    Optimized for A100 80GB
    """
    def __init__(self, data_dir, metadata_file='metadata.json', 
                 num_frames=None, size=None, transform=None):
        """
        Args:
            data_dir: Directory containing processed videos and metadata
            metadata_file: Name of metadata JSON file
            num_frames: Override number of frames (None = use from metadata)
            size: Override size (None = use from metadata)
            transform: Optional transform function
        """
        self.data_dir = Path(data_dir)
        
        # Load metadata
        metadata_path = self.data_dir / metadata_file
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.num_frames = num_frames
        self.size = size
        self.transform = transform
        
        print(f"Loaded GooglePhotosVideoDataset:")
        print(f"  Videos: {len(self.metadata)}")
        print(f"  Data dir: {data_dir}")
        
        if self.metadata:
            sample = self.metadata[0]
            print(f"  Resolution: {sample['resolution']}")
            print(f"  FPS: {sample['fps']}")
            print(f"  Frame range: {min(m['num_frames'] for m in self.metadata)}-{max(m['num_frames'] for m in self.metadata)}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load video frames
        frames = self.load_video_frames(item['processed_path'])
        
        # Convert to tensor [T, C, H, W]
        frames_tensor = torch.from_numpy(frames).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        
        # Normalize to [-1, 1] (standard for diffusion models)
        frames_tensor = (frames_tensor / 127.5) - 1.0
        
        # Apply transform if provided
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        
        return {
            'video': frames_tensor,
            'video_id': item['video_id'],
            'num_frames': frames_tensor.shape[0],
            'text': self.generate_caption(item),  # Simple caption generation
            'metadata': item,
        }
    
    def load_video_frames(self, video_path):
        """Load video frames as numpy array [T, H, W, C]"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        target_frames = self.num_frames if self.num_frames else 999999
        
        while len(frames) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if self.size and tuple(frame_rgb.shape[:2]) != self.size:
                frame_rgb = cv2.resize(frame_rgb, self.size, interpolation=cv2.INTER_LANCZOS4)
            
            frames.append(frame_rgb)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")
        
        # Pad with last frame if needed
        if self.num_frames and len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        
        return np.array(frames)
    
    def generate_caption(self, item):
        """
        Generate a simple caption for the video
        TODO: Replace with proper captioning model (BLIP-2, LLaVA, etc.)
        """
        # For now, just use a generic caption
        # You can enhance this later with actual video captioning
        duration = item['duration_seconds']
        return f"A {duration:.1f} second video from my personal collection"

# Test the dataset
if __name__ == "__main__":
    dataset = GooglePhotosVideoDataset('../google_photos_processed')
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"  Video shape: {sample['video'].shape}")
        print(f"  Video range: [{sample['video'].min():.2f}, {sample['video'].max():.2f}]")
        print(f"  Caption: {sample['text']}")
        print(f"  Video ID: {sample['video_id']}")
EOF

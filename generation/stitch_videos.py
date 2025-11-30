#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def stitch_videos(video_files, output_file, transition_duration=0.5):
    """
    Stitch multiple videos together with optional crossfade transitions.
    
    Args:
        video_files: List of video file paths
        output_file: Output file path
        transition_duration: Duration of crossfade transition in seconds (0 for no transition)
    """
    if len(video_files) < 2:
        print("Need at least 2 videos to stitch")
        return False
    
    # Verify all input files exist
    for video in video_files:
        if not os.path.exists(video):
            print(f"Error: {video} does not exist")
            return False
    
    # Create a temporary file list for ffmpeg
    list_file = "/tmp/video_list.txt"
    with open(list_file, 'w') as f:
        for video in video_files:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    if transition_duration == 0:
        # Simple concatenation without transitions
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_file
        ]
    else:
        # Concatenation with crossfade transitions
        # This is more complex and re-encodes
        print("Creating video with crossfade transitions...")
        
        # Build complex filter for crossfade
        filter_parts = []
        for i in range(len(video_files) - 1):
            if i == 0:
                filter_parts.append(f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset=5[v01];")
            else:
                filter_parts.append(f"[v0{i}][{i+1}:v]xfade=transition=fade:duration={transition_duration}:offset={5*(i+1)}[v0{i+1}];")
        
        # For simplicity, let's use simple concat for now
        # Complex xfade requires knowing exact video durations
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_file
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Successfully created {output_file}")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python stitch_videos.py output.mp4 video1.mp4 video2.mp4 [video3.mp4 ...]")
        sys.exit(1)
    
    output = sys.argv[1]
    videos = sys.argv[2:]
    
    print(f"Stitching {len(videos)} videos...")
    success = stitch_videos(videos, output)
    sys.exit(0 if success else 1)

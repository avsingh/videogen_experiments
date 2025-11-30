#!/usr/bin/env python3
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

def extract_last_frame(video_path, output_image_path):
    """Extract the last frame from a video."""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', 'select=eq(n\\,0)',  # Get last frame
        '-vframes', '1',
        '-update', '1',
        output_image_path
    ]
    
    # Better approach: get duration and extract frame at end
    cmd = [
        'ffmpeg', '-y',
        '-sseof', '-1',  # Seek to 1 second before end
        '-i', video_path,
        '-update', '1',
        '-q:v', '1',  # Best quality
        output_image_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def generate_video(image_path, prompt, output_name):
    """Generate a video using the Wan model."""
    cmd = [
        'python', 'generate.py',
        '--task', 'ti2v-5B',
        '--size', '1280*704',
        '--ckpt_dir', './Wan2.2-TI2V-5B',
        '--image', image_path,
        '--prompt', prompt
    ]
    
    print(f"Generating video from {image_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating video: {result.stderr}")
        return None
    
    # Find the most recently created video
    video_files = list(Path('.').glob('ti2v-5B_*.mp4'))
    if not video_files:
        return None
    
    latest_video = max(video_files, key=os.path.getctime)
    return str(latest_video)

def stitch_videos(video_list, output_path):
    """Stitch multiple videos together."""
    list_file = '/tmp/video_chain_list.txt'
    with open(list_file, 'w') as f:
        for video in video_list:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def chain_generate(initial_image, prompt, num_iterations, output_dir='chain_output'):
    """
    Generate a chain of videos, using the last frame of each as input for the next.
    
    Args:
        initial_image: Path to starting image
        prompt: Text prompt for all generations
        num_iterations: Number of videos to generate in the chain
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_videos = []
    current_image = initial_image
    
    print(f"\n{'='*60}")
    print(f"Starting chain generation:")
    print(f"  Initial image: {initial_image}")
    print(f"  Prompt: {prompt}")
    print(f"  Iterations: {num_iterations}")
    print(f"{'='*60}\n")
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        # Generate video
        video_path = generate_video(current_image, prompt, f"chain_{i}")
        
        if not video_path:
            print(f"Failed to generate video at iteration {i+1}")
            break
        
        # Copy to output directory with sequential name
        output_video = os.path.join(output_dir, f"segment_{i:03d}.mp4")
        os.rename(video_path, output_video)
        generated_videos.append(output_video)
        
        print(f"✓ Generated: {output_video}")
        
        # Extract last frame for next iteration (unless this is the last iteration)
        if i < num_iterations - 1:
            next_image = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            if extract_last_frame(output_video, next_image):
                current_image = next_image
                print(f"✓ Extracted frame: {next_image}")
            else:
                print("Failed to extract last frame")
                break
    
    # Stitch all videos together
    if len(generated_videos) > 1:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_output = os.path.join(output_dir, f'chained_video_{timestamp}.mp4')
        
        print(f"\n{'='*60}")
        print(f"Stitching {len(generated_videos)} videos together...")
        
        if stitch_videos(generated_videos, final_output):
            print(f"✓ Final video created: {final_output}")
            return final_output
        else:
            print("Failed to stitch videos")
            return None
    elif len(generated_videos) == 1:
        return generated_videos[0]
    else:
        return None

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python chain_generate.py <initial_image> <prompt> <num_iterations>")
        print("\nExample:")
        print("  python chain_generate.py sunset.jpg 'golden hour sunset over ocean' 5")
        sys.exit(1)
    
    initial_image = sys.argv[1]
    prompt = sys.argv[2]
    num_iterations = int(sys.argv[3])
    
    if not os.path.exists(initial_image):
        print(f"Error: Image {initial_image} not found")
        sys.exit(1)
    
    if num_iterations < 1:
        print("Error: num_iterations must be at least 1")
        sys.exit(1)
    
    final_video = chain_generate(initial_image, prompt, num_iterations)
    
    if final_video:
        print(f"\n{'='*60}")
        print(f"SUCCESS! Final video: {final_video}")
        print(f"{'='*60}\n")
    else:
        print("\nChain generation failed")
        sys.exit(1)

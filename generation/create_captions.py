import os

# Get all image files
images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

# Caption template for sunset style
caption_template = "cinematic sunset lighting, golden hour, warm orange and pink tones, dramatic sky, soft glowing light, beautiful sunset atmosphere"

for img in images:
    # Create corresponding .txt file
    txt_file = os.path.splitext(img)[0] + '.txt'
    
    # Skip if caption already exists
    if os.path.exists(txt_file):
        print(f"Skipping {txt_file} - already exists")
        continue
    
    # Write caption
    with open(txt_file, 'w') as f:
        f.write(caption_template)
    
    print(f"Created caption for {img}")

print(f"\nTotal images: {len(images)}")

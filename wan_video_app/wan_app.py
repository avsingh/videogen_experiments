import os
import subprocess
from flask import Flask, render_template, request, send_file, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import glob
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)  # For session management

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Simple user credentials - CHANGE THESE!
USERS = {
    'admin': 'scrypt:32768:8:1$QBdvhTalvDoPaCeS$7ba08a4495b91dd2fbc1d90461803b737164f02f6ad9948ad39e8dc1c007538c9d3950462afdef5fff8cea2a5b8f949210a80c7cca1887729db221441faa1167'
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and check_password_hash(USERS[username], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

@app.route('/generate', methods=['POST'])
@login_required
def generate_video():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        prompt = request.form.get('prompt', '')
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, GIF, or WEBP'}), 400
        
        if not prompt:
            return jsonify({'error': 'Please provide a prompt'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run the generation command
        cmd = [
            'python', 'generate.py',
            '--task', 'ti2v-5B',
            '--size', '1280*704',
            '--ckpt_dir', './Wan2.2-TI2V-5B',
            '--image', filepath,
            '--prompt', prompt
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Generation failed: {result.stderr}'}), 500
        
        # Find the most recent generated video
        video_files = glob.glob('ti2v-5B_*.mp4')
        if not video_files:
            return jsonify({'error': 'No video file generated'}), 500
        
        latest_video = max(video_files, key=os.path.getctime)
        
        return jsonify({
            'success': True,
            'video_path': latest_video,
            'message': 'Video generated successfully!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-chain', methods=['POST'])
@login_required
def generate_chain():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        prompt = request.form.get('prompt', '')
        iterations = int(request.form.get('iterations', 3))
        
        if file.filename == '' or not prompt:
            return jsonify({'error': 'Image and prompt required'}), 400
        
        if iterations < 1 or iterations > 10:
            return jsonify({'error': 'Iterations must be between 1 and 10'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Create output directory
        output_dir = f'chain_output_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        generated_videos = []
        current_image = filepath
        
        # Generate chain
        for i in range(iterations):
            # Generate video
            cmd = [
                'python', 'generate.py',
                '--task', 'ti2v-5B',
                '--size', '1280*704',
                '--ckpt_dir', './Wan2.2-TI2V-5B',
                '--image', current_image,
                '--prompt', prompt
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return jsonify({'error': f'Generation failed at iteration {i+1}: {result.stderr}'}), 500
            
            # Find latest video
            video_files = glob.glob('ti2v-5B_*.mp4')
            if not video_files:
                return jsonify({'error': f'No video generated at iteration {i+1}'}), 500
            
            latest_video = max(video_files, key=os.path.getctime)
            segment_path = os.path.join(output_dir, f'segment_{i:03d}.mp4')
            os.rename(latest_video, segment_path)
            generated_videos.append(segment_path)
            
            # Extract last frame for next iteration
            if i < iterations - 1:
                next_image = os.path.join(output_dir, f'frame_{i:03d}.jpg')
                extract_cmd = [
                    'ffmpeg', '-y',
                    '-sseof', '-1',
                    '-i', segment_path,
                    '-update', '1',
                    '-q:v', '1',
                    next_image
                ]
                subprocess.run(extract_cmd, capture_output=True)
                current_image = next_image
        
        # Stitch videos
        final_output = f'chained_{timestamp}.mp4'
        list_file = '/tmp/chain_list.txt'
        
        with open(list_file, 'w') as f:
            for video in generated_videos:
                f.write(f"file '{os.path.abspath(video)}'\n")
        
        stitch_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            final_output
        ]
        
        result = subprocess.run(stitch_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Stitching failed: {result.stderr}'}), 500
        
        return jsonify({
            'success': True,
            'video_path': final_output,
            'segments': iterations,
            'message': f'Generated {iterations}-segment chained video successfully!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stitch', methods=['POST'])
@login_required
def stitch_videos():
    try:
        data = request.get_json()
        video_files = data.get('videos', [])
        
        if len(video_files) < 2:
            return jsonify({'error': 'Need at least 2 videos to stitch'}), 400
        
        # Verify all files exist
        for video in video_files:
            if not os.path.exists(video):
                return jsonify({'error': f'Video {video} not found'}), 404
        
        # Create output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'stitched_{timestamp}.mp4'
        
        # Create file list for ffmpeg
        list_file = '/tmp/video_list.txt'
        with open(list_file, 'w') as f:
            for video in video_files:
                f.write(f"file '{os.path.abspath(video)}'\n")
        
        # Run ffmpeg to concatenate
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Stitching failed: {result.stderr}'}), 500
        
        return jsonify({
            'success': True,
            'video_path': output_file,
            'message': f'Successfully stitched {len(video_files)} videos!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
@login_required
def download_video(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/video/<path:filename>')
@login_required
def serve_video(filename):
    try:
        return send_file(filename, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/history')
@login_required
def get_history():
    try:
        # Get all video files (regular, stitched, and chained)
        video_patterns = ['ti2v-5B_*.mp4', 'stitched_*.mp4', 'chained_*.mp4']
        video_files = []
        for pattern in video_patterns:
            video_files.extend(glob.glob(pattern))
        
        videos = []
        for video in sorted(video_files, key=os.path.getctime, reverse=True)[:20]:
            videos.append({
                'filename': video,
                'created': datetime.fromtimestamp(os.path.getctime(video)).strftime('%Y-%m-%d %H:%M:%S')
            })
        return jsonify({'videos': videos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

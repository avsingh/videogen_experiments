# Wan Video Generator - Deployment with Authentication

## Files to upload to Lambda:
1. wan_app.py (main Flask app with authentication)
2. templates/index.html (main interface)
3. templates/login.html (login page)
4. generate_password.py (password setup helper)

## Setup Instructions:

### 1. Upload files to Lambda machine

From your Mac terminal:
```bash
cd ~/Wan2.2
scp -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem wan_app.py ubuntu@209.20.156.222:~/Wan2.2/
scp -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem templates/login.html ubuntu@209.20.156.222:~/Wan2.2/templates/
scp -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem templates/index.html ubuntu@209.20.156.222:~/Wan2.2/templates/
scp -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem generate_password.py ubuntu@209.20.156.222:~/Wan2.2/
```

### 2. Set your password (IMPORTANT!)

SSH into Lambda:
```bash
ssh -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem ubuntu@209.20.156.222
```

Run the password generator:
```bash
cd ~/Wan2.2
python generate_password.py
```

This will generate a password hash. Copy it and edit wan_app.py:
```bash
nano wan_app.py
```

Find this section (around line 18):
```python
USERS = {
    'admin': generate_password_hash('change_this_password_123')
}
```

Replace with your username and password hash:
```python
USERS = {
    'your_username': 'your_password_hash_from_generator'
}
```

Save and exit (Ctrl+X, then Y, then Enter)

### 3. Run the app

```bash
python wan_app.py
```

### 4. Access the app

From your Mac, open a new terminal and create SSH tunnel:
```bash
ssh -i /Users/av/src/wan-fine-tune/wan-finetune-key.pem -L 5000:localhost:5000 ubuntu@209.20.156.222
```

Open browser: http://localhost:5000

Login with your credentials!

### 5. Optional: Open firewall for public access

If you configured Lambda firewall to allow port 5000, anyone can access:
http://209.20.156.222:5000

But they'll need your username and password to use the app!

## Security Notes:

- Default credentials are admin/change_this_password_123 - CHANGE THESE!
- Use a strong password (at least 8 characters)
- You can add multiple users by adding more entries to the USERS dictionary
- Sessions expire when you close the browser
- All routes are protected - no one can use the app without logging in

## Adding More Users:

Run the password generator for each user:
```bash
python generate_password.py
```

Then add to USERS dictionary:
```python
USERS = {
    'admin': 'hash1',
    'user2': 'hash2',
    'user3': 'hash3'
}
```

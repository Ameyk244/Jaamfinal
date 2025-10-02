import os

# --- Folders to create ---
folders = ["uploads", "results"]

# --- Files to create with content ---
files_content = {
    ".env": "HF_TOKEN=your_huggingface_token\nNGROK_AUTH_TOKEN=your_ngrok_token\n",
    ".gitignore": ".env\n__pycache__/\nuploads/\nresults/\n",
    "requirements.txt": """tensorflow
matplotlib
huggingface-hub
flask
pyngrok
pillow
opencv-python
python-dotenv
"""
}

# --- Create folders ---
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# --- Create files if they don’t exist ---
for filename, content in files_content.items():
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(content)

print("✅ Project setup completed! Folders & config files created.")
"""
Setup Script to Download Models from Google Drive Automatically
Run this ONCE before starting the backend, or during the Docker build process.
"""

import os
import gdown
from pathlib import Path

print("=" * 60)
print("🚀 AgriSurgeon Model Download Setup")
print("=" * 60)

# Google Drive folder ID (shared folder with models)
GOOGLE_DRIVE_FOLDER_ID = "1AxTTvtdAscl_KKPQRZOmGWAroWc6PNnG"

# Models directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

print(f"\n📁 Models directory: {MODELS_DIR.absolute()}")

# Files to download (filename: Google Drive file ID)
# NOTE: Ensure these IDs match the actual file IDs in your Google Drive folder.
MODEL_FILES = {
    "MobileNetV3_SOTA_Regularized.h5": "1g2k3h4j5k6l7m8n9o0p1q2r3s4t5u6v7",
    "svm_hybrid_model.pkl": "1a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "scaler.pkl": "1x9y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4",
    "class_indices.json": "1m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0"
}

print("\n📦 Checking Required Files...")

# Check if all files exist
all_files_exist = all((MODELS_DIR / f).exists() for f in MODEL_FILES.keys())

if all_files_exist:
    print("\n✅ All model files found locally! No download needed.")
    for filename in MODEL_FILES.keys():
        print(f"  • {filename:40} ✓ EXISTS")
else:
    print("\n⚠️ Downloading missing files from Google Drive...")
    for filename, file_id in MODEL_FILES.items():
        filepath = MODELS_DIR / filename
        
        if not filepath.exists():
            print(f"\n⬇️ Downloading {filename}...")
            # gdown uses the direct file ID for downloads
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, str(filepath), quiet=False)
        else:
            print(f"✅ {filename} already exists. Skipping.")

# Check for deployment config
deployment_config = Path(__file__).parent / "deployment_config.json"
if deployment_config.exists():
    print(f"\n✓ deployment_config.json found")
else:
    print(f"\n✗ deployment_config.json missing (will use default settings if applicable)")

print("\n" + "=" * 60)
print("✅ Setup complete!")
print("=" * 60)

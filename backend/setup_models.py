"""
Setup Script to Download Models from Google Drive
Run this ONCE before starting the backend
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
MODEL_FILES = {
    "MobileNetV3_SOTA_Regularized.h5": "1g2k3h4j5k6l7m8n9o0p1q2r3s4t5u6v7",
    "svm_hybrid_model.pkl": "1a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "scaler.pkl": "1x9y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4",
    "class_indices.json": "1m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0"
}

print("\n📦 Required Files:")
for filename in MODEL_FILES.keys():
    filepath = MODELS_DIR / filename
    status = "✓ EXISTS" if filepath.exists() else "✗ MISSING"
    print(f"  • {filename:40} {status}")

# Check if all files exist
all_files_exist = all((MODELS_DIR / f).exists() for f in MODEL_FILES.keys())

if all_files_exist:
    print("\n✅ All model files found locally!")
    print("No download needed.")
else:
    print("\n⚠️  Some files are missing. Download from Google Drive:")
    print(f"📌 Folder Link: https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}")
    print("\nManual Steps:")
    print("1. Open the Google Drive folder link above")
    print("2. Download these files:")
    for filename in MODEL_FILES.keys():
        print(f"   • {filename}")
    print(f"3. Place them in: {MODELS_DIR.absolute()}")
    print("\nOR use gdown to download automatically:")
    print("  pip install gdown")
    print(f"  gdown --folder-id {GOOGLE_DRIVE_FOLDER_ID} -O {MODELS_DIR}")

# Deployment config already exists locally
deployment_config = Path(__file__).parent / "deployment_config.json"
if deployment_config.exists():
    print(f"\n✓ deployment_config.json found")
else:
    print(f"\n✗ deployment_config.json missing")

print("\n" + "=" * 60)
print("✅ Setup guide complete!")
print("=" * 60)

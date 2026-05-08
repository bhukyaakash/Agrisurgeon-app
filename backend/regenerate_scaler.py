"""
Regenerate scaler.pkl file
This script creates a fresh scaler file for the backend
"""

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

print("🔧 Regenerating scaler.pkl...\n")

# Create models directory if it doesn't exist
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

# Create realistic temperature and humidity data
dummy_env_data = np.array([
    [10, 20],   # Cold, dry
    [15, 40],   # Cool, moderate
    [20, 60],   # Mild, humid
    [25, 75],   # Warm, very humid
    [30, 85],   # Hot, very humid
    [35, 90],   # Very hot, extreme humidity
    [40, 95],   # Extreme heat, extreme humidity
    [45, 100],  # Maximum conditions
])

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(dummy_env_data)

# Save scaler
scaler_path = os.path.join(models_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler regenerated successfully!")
print(f"📁 Location: {os.path.abspath(scaler_path)}")
print(f"\n✓ Mean: {scaler.mean_}")
print(f"✓ Scale: {scaler.scale_}")

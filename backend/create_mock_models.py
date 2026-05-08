"""
Create mock ML models for testing without real models
Run this ONCE to generate placeholder models
"""

import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras

# Create models directory
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

print("🔧 Creating mock ML models for testing...")

# ============================================
# 1. Create Mock CNN Model (MobileNetV3)
# ============================================
print("  • Creating mock CNN model...")

# Simple sequential model to mimic MobileNetV3
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1280, activation='relu'),  # Feature extraction layer
    keras.layers.Dense(38, activation='softmax')  # 38 diseases
])

model_path = os.path.join(models_dir, "MobileNetV3_SOTA_Regularized.h5")
model.save(model_path)
print(f"    ✓ Saved: {model_path}")

# ============================================
# 2. Create Mock Scaler
# ============================================
print("  • Creating mock scaler...")

# Create dummy data for scaler (temperature, humidity)
dummy_env_data = np.array([
    [10, 0],
    [20, 50],
    [30, 100],
    [50, 100]
])

scaler = StandardScaler()
scaler.fit(dummy_env_data)

scaler_path = os.path.join(models_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"    ✓ Saved: {scaler_path}")

# ============================================
# 3. Create Mock SVM Model
# ============================================
print("  • Creating mock SVM model...")

# Create dummy features (1280 from CNN + 2 from environment = 1282)
dummy_features = np.random.randn(100, 1282)
dummy_labels = np.random.randint(0, 38, 100)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(dummy_features, dummy_labels)

svm_path = os.path.join(models_dir, "svm_hybrid_model.pkl")
with open(svm_path, 'wb') as f:
    pickle.dump(svm_model, f)
print(f"    ✓ Saved: {svm_path}")

# ============================================
# 4. Create Class Indices
# ============================================
print("  • Creating class indices...")

diseases = [
    "Apple Scab", "Apple Rust", "Apple Healthy",
    "Tomato Late Blight", "Tomato Early Blight", "Tomato Mosaic Virus", "Tomato Healthy",
    "Potato Late Blight", "Potato Early Blight", "Potato Healthy",
    "Cherry Powdery Mildew", "Cherry Healthy",
    "Grape Black Rot", "Grape Leaf Blight", "Grape Healthy",
    "Corn Leaf Blight", "Corn Healthy",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Blueberry Rust", "Blueberry Healthy",
    "Citrus Black Spot", "Citrus Canker", "Citrus Healthy",
    "Peach Leaf Curl", "Peach Healthy",
    "Bell Pepper Anthracnose", "Bell Pepper Bacterial Spot", "Bell Pepper Healthy",
    "Cucumber Downy Mildew", "Cucumber Healthy",
    "Lettuce Bottom Rot", "Lettuce Healthy",
    "Rice Blast", "Rice Healthy",
    "Wheat Septoria", "Wheat Healthy"
]

class_indices = {disease: idx for idx, disease in enumerate(diseases)}

indices_path = os.path.join(models_dir, "class_indices.json")
with open(indices_path, 'w') as f:
    json.dump(class_indices, f, indent=2)
print(f"    ✓ Saved: {indices_path}")

print("\n✅ Mock models created successfully!")
print(f"📁 Models location: {os.path.abspath(models_dir)}/")
print("\nModels created:")
print(f"  • {model_path}")
print(f"  • {scaler_path}")
print(f"  • {svm_path}")
print(f"  • {indices_path}")
print("\n⚠️  Note: These are mock models for TESTING ONLY")
print("Replace them with your real trained models when ready!")

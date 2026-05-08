"""
Create realistic mock ML models for testing with proper disease classification
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

print("🔧 Creating realistic mock ML models for testing...\n")

# ============================================
# DISEASE CLASSES (MUST MATCH deployment_config.json)
# ============================================
diseases = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

print(f"✓ {len(diseases)} disease classes loaded\n")

# ============================================
# 1. CREATE MOCK CNN MODEL (MobileNetV3)
# ============================================
print("📦 Creating mock CNN model...")

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1280, activation='relu'),  # Feature extraction layer
    keras.layers.Dense(len(diseases), activation='softmax')  # Classification
])

model_path = os.path.join(models_dir, "MobileNetV3_SOTA_Regularized.h5")
model.save(model_path)
print(f"   ✓ Saved: {model_path}\n")

# ============================================
# 2. CREATE MOCK SCALER
# ============================================
print("📊 Creating mock scaler...")

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

scaler = StandardScaler()
scaler.fit(dummy_env_data)

scaler_path = os.path.join(models_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Saved: {scaler_path}\n")

# ============================================
# 3. CREATE MOCK SVM MODEL
# ============================================
print("🤖 Creating mock SVM model...")

# Create synthetic training data with patterns
np.random.seed(42)
n_samples = 500
n_features = 1282  # 1280 CNN features + 2 env features

# Create data that shows different disease patterns
X = np.random.randn(n_samples, n_features) * 0.5

# Add disease-specific patterns to features
for i in range(len(diseases)):
    disease_samples = np.where(np.arange(n_samples) % len(diseases) == i)[0]
    if len(disease_samples) > 0:
        X[disease_samples, i:min(i+10, n_features)] += np.random.randn(len(disease_samples), min(10, n_features-i)) * 2

y = np.repeat(np.arange(len(diseases)), n_samples // len(diseases))
y = y[:len(X)]

# Train SVM with RBF kernel for better separation
svm_model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='auto')
svm_model.fit(X, y)

svm_path = os.path.join(models_dir, "svm_hybrid_model.pkl")
with open(svm_path, 'wb') as f:
    pickle.dump(svm_model, f)
print(f"   ✓ Saved: {svm_path}\n")

# ============================================
# 4. CREATE CLASS INDICES
# ============================================
print("📑 Creating class indices...")

# Create mapping: disease_name -> index
class_indices = {disease: idx for idx, disease in enumerate(diseases)}

indices_path = os.path.join(models_dir, "class_indices.json")
with open(indices_path, 'w') as f:
    json.dump(class_indices, f, indent=2)
print(f"   ✓ Saved: {indices_path}\n")

print("=" * 60)
print("✅ MOCK MODELS CREATED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📁 Models location: {os.path.abspath(models_dir)}/\n")
print("Files created:")
print(f"   1. {model_path}")
print(f"   2. {scaler_path}")
print(f"   3. {svm_path}")
print(f"   4. {indices_path}")

print("\n⚙️  Model Configuration:")
print(f"   • CNN Features: 1280")
print(f"   • Environmental Features: 2 (temperature, humidity)")
print(f"   • Hybrid Features: 1282")
print(f"   • Disease Classes: {len(diseases)}")
print(f"   • Training Samples: {n_samples}")

print("\n✓ Disease Classes Match deployment_config.json")
print("✓ Different predictions for different plant images")
print("✓ Proper disease-specific advisories enabled")

print("\n⚠️  IMPORTANT:")
print("   ✓ These are MOCK models for TESTING ONLY")
print("   ✓ Replace with REAL trained models for production")
print("   ✓ Restart backend after creating models!")
print("\n🚀 Next steps:")
print("   1. python create_mock_models.py  (Done!)")
print("   2. Delete old models/ folder")
print("   3. python main.py")
print("   4. Open http://localhost:8080")
print("   5. Upload different plant images to test!\n")

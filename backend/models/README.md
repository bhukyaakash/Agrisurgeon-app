# AgriSurgeon ML Models Directory

This directory contains the pre-trained machine learning models required for AgriSurgeon to function.

## Required Model Files

### 1. **MobileNetV3_SOTA_Regularized.h5** (45 MB)
- **Type**: Deep Convolutional Neural Network (CNN)
- **Architecture**: MobileNetV3-Large with ImageNet pre-training
- **Purpose**: Extract 1280-dimensional visual features from leaf images
- **Parameters**: 5.4 million
- **Input**: RGB images (224×224×3)
- **Output**: Feature vector (1280-dim)
- **Framework**: TensorFlow/Keras
- **Training Data**: 61,519 images from New Plant Diseases Dataset
- **Regularization**: L2 weight decay (λ=0.01) + Dropout (p=0.5)
- **Training Accuracy**: ~99.1%
- **Validation Accuracy**: ~84.32%

### 2. **scaler.pkl** (1 KB)
- **Type**: StandardScaler from scikit-learn
- **Purpose**: Normalize environmental sensor data (temperature, humidity)
- **Parameters**:
  - Temperature mean (μ_T): 26.3°C
  - Temperature std (σ_T): 8.1°C
  - Humidity mean (μ_H): 68.2%
  - Humidity std (σ_H): 22.4%
- **Format**: Python pickle
- **Used for**: Z-score normalization of environmental features

### 3. **svm_hybrid_model.pkl** (25 MB)
- **Type**: Linear Support Vector Machine with One-vs-Rest multiclass strategy
- **Purpose**: Classify disease from 1282-dimensional hybrid features
- **Classifiers**: 38 binary classifiers (one per disease category)
- **Kernel**: Linear
- **Regularization Parameter**: C = 0.1
- **Input Dimension**: 1282 (1280 visual + 2 environmental)
- **Classes**: 38 disease categories across 14 plant species
- **Training Samples**: ~43,066 (70% of 61,519)
- **Format**: Python pickle

### 4. **class_indices.json** (1 KB)
- **Type**: JSON mapping file
- **Purpose**: Bidirectional mapping between disease names and class indices
- **Format**:
  ```json
  {
    "Apple Scab": 0,
    "Apple Rust": 1,
    ...,
    "Tomato Mosaic Virus": 37
  }
  ```
- **Total Classes**: 38

## Disease Categories (38 Total)

### Apple (3)
- Apple Scab
- Apple Rust
- Apple Healthy

### Tomato (4)
- Tomato Late Blight
- Tomato Early Blight
- Tomato Mosaic Virus
- Tomato Healthy

### Potato (3)
- Potato Late Blight
- Potato Early Blight
- Potato Healthy

### Cherry (2)
- Cherry Powdery Mildew
- Cherry Healthy

### Grape (2)
- Grape Black Rot
- Grape Healthy

### Corn (2)
- Corn Leaf Blight
- Corn Healthy

### Other Species (22)
- Blueberry, Orange, Peach, Pepper, Raspberry, Soybean, Squash, Strawberry
- Multiple diseases per species + Healthy

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.50% |
| **Precision** | 98.51% |
| **Recall** | 98.50% |
| **F1-Score** | 98.50% |
| **Inference Time (CPU)** | 85-90 ms |
| **Inference Time (GPU)** | 12-14 ms |
| **Model Size** | ~70 MB (all files) |
| **Memory Requirement** | ~120 MB |

## Setup Instructions

### Step 1: Download Model Files

Download the four required files and place them in this directory (`backend/models/`):

```bash
cd backend/models/

# Verify directory structure
ls -lh
```

Expected output:
```
MobileNetV3_SOTA_Regularized.h5  45M  Jan 1 2024
scaler.pkl                         1K  Jan 1 2024
svm_hybrid_model.pkl              25M  Jan 1 2024
class_indices.json                 1K  Jan 1 2024
README.md                          XX  Jan 1 2024
```

### Step 2: Verify Model Integrity

```bash
# Check file sizes (should match above)
du -h *.h5 *.pkl *.json

# Verify HDF5 file validity
python3 -c "import h5py; f = h5py.File('MobileNetV3_SOTA_Regularized.h5', 'r'); print(f.keys()); f.close()"

# Test pickle files
python3 -c "import pickle; pickle.load(open('scaler.pkl', 'rb')); print('scaler.pkl OK')"
python3 -c "import pickle; pickle.load(open('svm_hybrid_model.pkl', 'rb')); print('svm_hybrid_model.pkl OK')"

# Verify JSON
python3 -c "import json; json.load(open('class_indices.json')); print('class_indices.json OK')"
```

### Step 3: Test Model Loading

```bash
cd ../../
python3 << 'EOF'
from backend.config import Config
from backend.main import model_manager

try:
    model_manager.load_models()
    print("✅ All models loaded successfully!")
    print(f"Diseases supported: {len(model_manager.disease_names)}")
    print(f"Disease list: {list(model_manager.disease_names.values())[:5]}...")
except Exception as e:
    print(f"❌ Error loading models: {e}")
EOF
```

## Deployment

### Using Docker

```bash
# Build Docker image
docker build -t agrisurgeon:latest -f docker/Dockerfile .

# Run container with model mounting
docker run -v $(pwd)/backend/models:/app/backend/models:ro \
           -p 8000:8000 \
           agrisurgeon:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up --build

# Check service health
curl http://localhost:8000/api/health
```

### Direct Execution

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Testing

### Health Check

```bash
curl -X GET http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00",
  "models_loaded": true
}
```

### Disease Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@leaf_image.jpg" \
  -F "temperature=22" \
  -F "humidity=75"
```

### Get All Diseases

```bash
curl -X GET http://localhost:8000/api/diseases
```

## Model Architecture Details

### Hybrid Feature Fusion

```
Input Image (224×224×3)
    ↓
[MobileNetV3-Large]
    ↓
Visual Features (1280-dim)
    ↓
  [Late Fusion] ← Environmental Data (2-dim)
    ↓
Hybrid Features (1282-dim)
    ↓
[Linear SVM × 38 Classes]
    ↓
Disease Prediction + Confidence
```

### Environmental Context

The scaler normalizes:
- **Temperature**: Input range [10-50]°C → Output z-score
- **Humidity**: Input range [0-100]% → Output z-score

This ensures environmental features don't dominate the SVM decision boundary.

## Security Considerations

1. **Model Files**: Mount as read-only in Docker (`volumes: ...models:/app/models:ro`)
2. **Permissions**: Ensure proper file permissions (owner: agrisurgeon user)
3. **Size Limits**: Maximum upload file size: 10 MB
4. **Inference Timeout**: 30 seconds per request

## Troubleshooting

### Models Not Loading

```bash
# Check file permissions
ls -la *.h5 *.pkl *.json

# Verify TensorFlow installation
python3 -c "import tensorflow as tf; print(tf.__version__)"

# Check HDF5 file corruption
python3 << 'EOF'
import tensorflow as tf
try:
    model = tf.keras.models.load_model('MobileNetV3_SOTA_Regularized.h5')
    print("✅ Model loads successfully")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

### Memory Issues

```bash
# Monitor memory during inference
watch -n 1 'ps aux | grep uvicorn | grep -v grep'

# Set memory limits
docker run -m 2g agrisurgeon:latest
```

### Slow Inference

```bash
# Check if GPU is available
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Enable GPU in production (if available)
export CUDA_VISIBLE_DEVICES=0
```

## Updates and Versioning

- **Model Version**: 1.0.0
- **Training Date**: January 2024
- **TensorFlow Version**: 2.14.0
- **scikit-learn Version**: 1.3.2

For model updates, replace files in-place and restart the API service.

## References

- Dataset: New Plant Diseases Dataset (87,885 images)
- CNN: MobileNetV3-Large (Howard et al., ICCV 2019)
- Classifier: Linear SVM with One-vs-Rest
- Paper: AgriSurgeon - Hybrid MobileNetV3-SVM-IoT Fusion Framework

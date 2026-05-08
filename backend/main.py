"""
AgriSurgeon API - AI-Powered Plant Disease Detection
Hybrid MobileNetV3-SVM-IoT Fusion Framework
"""

import os
import json
import pickle
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

# TensorFlow & ML
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# FastAPI & Web
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

# Configuration
from config import Config

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# GLOBAL MODEL MANAGER
# ============================================
class ModelManager:
    """Singleton for loading and managing ML models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.is_loaded = False
        return cls._instance
    
    def load_models(self):
        """Load all ML models and advisory database"""
        try:
            logger.info("Loading ML models...")
            
            # Load CNN model
            logger.info(f"Loading CNN from: {Config.MODEL_PATH}")
            self.cnn_model = keras.models.load_model(Config.MODEL_PATH)
            logger.info("✓ CNN model loaded")
            
            # Create feature extractor (output from penultimate layer)
            self.feature_extractor = keras.Model(
                inputs=self.cnn_model.input,
                outputs=self.cnn_model.layers[-2].output
            )
            logger.info("✓ Feature extractor created")
            
            # Load scaler
            logger.info(f"Loading scaler from: {Config.SCALER_PATH}")
            with open(Config.SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("✓ StandardScaler loaded")
            
            # Load SVM model
            logger.info(f"Loading SVM from: {Config.SVM_PATH}")
            with open(Config.SVM_PATH, 'rb') as f:
                self.svm_model = pickle.load(f)
            logger.info("✓ SVM model loaded")
            
            # Load class indices
            logger.info(f"Loading class indices from: {Config.CLASS_INDICES_PATH}")
            with open(Config.CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            
            # Create reverse mapping (index -> disease name)
            self.disease_names = {v: k for k, v in class_indices.items()}
            logger.info(f"✓ Class indices loaded ({len(self.disease_names)} classes)")
            
            # Load advisory database from deployment config
            logger.info("Loading advisory database...")
            deployment_config_path = os.path.join(
                os.path.dirname(__file__),
                "deployment_config.json"
            )
            
            if os.path.exists(deployment_config_path):
                with open(deployment_config_path, 'r') as f:
                    config_data = json.load(f)
                    self.advisory_db = config_data.get('advisory', {})
                    logger.info(f"✓ Advisory database loaded ({len(self.advisory_db)} entries)")
            else:
                logger.warning(f"Advisory database not found at {deployment_config_path}")
                self.advisory_db = {}
            
            self.is_loaded = True
            logger.info("✅ All models loaded successfully!")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Error loading models: Model not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise

# ============================================
# INITIALIZE MODEL MANAGER
# ============================================
model_manager = ModelManager()

# ============================================
# FASTAPI LIFESPAN CONTEXT
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        model_manager.load_models()
        yield
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")
        raise
    # Shutdown
    finally:
        logger.info("Shutting down AgriSurgeon API")

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=Config.CORS_CREDENTIALS,
    allow_methods=Config.CORS_METHODS,
    allow_headers=Config.CORS_HEADERS,
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    temperature: float
    humidity: float
    environmental_risk: str
    advisory: dict
    timestamp: str
    model_version: str

# ============================================
# UTILITY FUNCTIONS
# ============================================
def validate_image(image_data: bytes) -> bool:
    """Validate image file"""
    try:
        if len(image_data) > Config.MAX_FILE_SIZE:
            return False
        
        img = Image.open(io.BytesIO(image_data))
        # Check if valid image
        img.verify()
        
        # Reopen after verify
        img = Image.open(io.BytesIO(image_data))
        
        # Check format
        if img.format.lower() not in Config.ALLOWED_FORMATS:
            return False
        
        return True
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False

def process_image(image_data: bytes) -> np.ndarray:
    """Process image for model input"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize((Config.IMG_SIZE, Config.IMG_SIZE))
        
        # Normalize to 0-1
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise

def calculate_risk_level(disease_name: str, temperature: float, humidity: float) -> str:
    """Calculate disease risk level based on disease and environmental conditions"""
    # Define critical diseases
    critical_diseases = [
        "Tomato___Late_blight",
        "Potato___Late_blight",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ]
    
    if disease_name in critical_diseases:
        return "critical"
    
    # Check for optimal disease conditions
    if "healthy" in disease_name.lower():
        return "none"
    
    # Environmental risk assessment
    if (20 <= temperature <= 28) and (60 <= humidity <= 85):
        return "high"
    elif (15 <= temperature <= 30) and (50 <= humidity <= 95):
        return "medium"
    else:
        return "low"

# ============================================
# ROOT ENDPOINTS
# ============================================
@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": Config.API_TITLE,
        "version": Config.API_VERSION,
        "status": "✅ Running",
        "models_loaded": model_manager.is_loaded,
        "description": Config.API_DESCRIPTION
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": model_manager.is_loaded,
        "model_accuracy": Config.MODEL_ACCURACY,
        "disease_categories": Config.DISEASE_CATEGORIES
    }

# ============================================
# PREDICTION ENDPOINT
# ============================================
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...)
):
    """
    Predict plant disease from image and environmental data
    
    Args:
        file: Leaf image (JPEG/PNG)
        temperature: Temperature in Celsius (10-50)
        humidity: Humidity percentage (0-100)
    
    Returns:
        Disease prediction with confidence and advisory
    """
    try:
        # Validate models loaded
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Service unavailable."
            )
        
        # Read and validate image
        image_data = await file.read()
        if not validate_image(image_data):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or file too large (max 10MB)"
            )
        
        # Process image
        image_array = process_image(image_data)
        
        # Extract visual features
        visual_features = model_manager.feature_extractor.predict(
            image_array, 
            verbose=0
        )[0]
        
        # Validate environmental data
        if not (10 <= temperature <= 50):
            raise HTTPException(
                status_code=400,
                detail="Temperature must be between 10-50°C"
            )
        if not (0 <= humidity <= 100):
            raise HTTPException(
                status_code=400,
                detail="Humidity must be between 0-100%"
            )
        
        # Normalize environmental data
        env_data = np.array([[temperature, humidity]])
        env_normalized = model_manager.scaler.transform(env_data)[0]
        
        # Create hybrid features
        hybrid_features = np.concatenate([visual_features, env_normalized])
        hybrid_features = hybrid_features.reshape(1, -1)
        
        # Predict disease
        disease_idx = model_manager.svm_model.predict(hybrid_features)[0]
        disease_name = model_manager.disease_names.get(disease_idx, "Unknown Disease")
        
        # Calculate confidence
        decision_scores = model_manager.svm_model.decision_function(hybrid_features)[0]
        confidence = float((np.max(decision_scores) + 1) / 2)
        confidence = min(max(confidence, 0.0), 1.0)
        
        # Get advisory from database (exact match or generic fallback)
        advisory = model_manager.advisory_db.get(
            disease_name,
            model_manager.advisory_db.get("generic", {
                "cause": "Unknown pathogen",
                "cure": "Consult agricultural specialist",
                "prevention": "Implement standard disease management practices",
                "pathogen_type": "Unknown",
                "severity": "Unknown"
            })
        )
        
        # Calculate risk level
        risk_level = calculate_risk_level(disease_name, temperature, humidity)
        
        logger.info(f"✓ Prediction: {disease_name} (confidence: {confidence:.2%})")
        
        return {
            "disease": disease_name,
            "confidence": round(confidence, 4),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "environmental_risk": risk_level,
            "advisory": advisory,
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "1.0.0"
        }
    
    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================
# INFORMATION ENDPOINTS
# ============================================
@app.get("/api/diseases")
async def get_diseases():
    """Get list of all diseases"""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    diseases = list(model_manager.disease_names.values())
    return {
        "count": len(diseases),
        "diseases": sorted(diseases)
    }

@app.get("/api/models")
async def get_models():
    """Get model information"""
    return {
        "cnn": {
            "name": "MobileNetV3",
            "architecture": "MobileNetV3-Small",
            "parameters": "5.4M",
            "input_size": (224, 224, 3),
            "purpose": "Visual feature extraction"
        },
        "svm": {
            "name": "Linear SVM",
            "kernel": "linear",
            "classes": Config.DISEASE_CATEGORIES,
            "input_features": Config.HYBRID_FEATURES_DIM
        },
        "performance": {
            "accuracy": Config.MODEL_ACCURACY,
            "precision": Config.MODEL_PRECISION,
            "recall": Config.MODEL_RECALL,
            "f1_score": Config.MODEL_F1_SCORE
        },
        "training": {
            "samples": Config.TRAINING_SAMPLES,
            "plant_species": Config.PLANT_SPECIES,
            "disease_categories": Config.DISEASE_CATEGORIES
        }
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get API statistics"""
    return {
        "api_version": Config.API_VERSION,
        "model_accuracy": f"{Config.MODEL_ACCURACY * 100:.2f}%",
        "supported_diseases": Config.DISEASE_CATEGORIES,
        "plant_species": Config.PLANT_SPECIES,
        "training_samples": Config.TRAINING_SAMPLES,
        "inference_timeout": f"{Config.INFERENCE_TIMEOUT}s",
        "max_image_size": f"{Config.MAX_FILE_SIZE / (1024*1024):.0f}MB",
        "supported_formats": list(Config.ALLOWED_FORMATS),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# STARTUP MESSAGE
# ============================================
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 AgriSurgeon API starting...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

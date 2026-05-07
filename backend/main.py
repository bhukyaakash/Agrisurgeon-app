"""
AgriSurgeon - Main FastAPI Application
AI-Powered Plant Disease Detection System
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pickle
from PIL import Image
import io

from config import Config

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# MODEL LOADING
# ============================================
class ModelManager:
    """Manages ML model lifecycle"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.scaler = None
        self.svm_model = None
        self.class_indices = None
        self.disease_names = None
        self.advisory_db = None
        self.is_loaded = False
    
    def load_models(self):
        """Load all required models"""
        try:
            logger.info("Loading ML models...")
            
            # Load CNN model
            if not os.path.exists(Config.MODEL_PATH):
                raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}")
            
            self.model = tf.keras.models.load_model(Config.MODEL_PATH)
            logger.info("✓ CNN model loaded")
            
            # Create feature extractor (remove softmax head)
            self.feature_extractor = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-2].output
            )
            logger.info("✓ Feature extractor created")
            
            # Load scaler
            with open(Config.SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("✓ StandardScaler loaded")
            
            # Load SVM model
            with open(Config.SVM_PATH, 'rb') as f:
                self.svm_model = pickle.load(f)
            logger.info("✓ SVM model loaded")
            
            # Load class indices
            with open(Config.CLASS_INDICES_PATH, 'r') as f:
                self.class_indices = json.load(f)
            self.disease_names = {v: k for k, v in self.class_indices.items()}
            logger.info("✓ Class indices loaded")
            
            # Load advisory database
            self.advisory_db = self._create_advisory_db()
            logger.info("✓ Advisory database created")
            
            self.is_loaded = True
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            raise
    
    def _create_advisory_db(self):
        """Create advisory database for all diseases"""
        return {
            "Apple Scab": {
                "cause": "Venturia inaequalis (fungal pathogen)",
                "cure": "Captan, Mancozeb fungicide applications every 7-14 days during growing season",
                "prevention": "Remove fallen leaves, improve air circulation, prune for better canopy airflow",
                "pathogen_type": "Fungal",
                "severity": "High"
            },
            "Apple Rust": {
                "cause": "Gymnosporangium species (fungal pathogen)",
                "cure": "Myclobutanil or Triadimefon fungicides applied at bud break",
                "prevention": "Remove alternate host plants (junipers), improve drainage",
                "pathogen_type": "Fungal",
                "severity": "Medium"
            },
            "Tomato Late Blight": {
                "cause": "Phytophthora infestans (oomycete pathogen)",
                "cure": "Metalaxyl, Dimethomorph systemic fungicides - apply every 5-7 days",
                "prevention": "Avoid overhead irrigation, ensure 90% humidity threshold not exceeded, remove infected leaves",
                "pathogen_type": "Fungal",
                "severity": "Critical"
            },
            "Tomato Early Blight": {
                "cause": "Alternaria solani (fungal pathogen)",
                "cure": "Chlorothalonil, Mancozeb fungicides applied preventatively",
                "prevention": "Remove lower leaves, improve air circulation, mulch soil to prevent splash",
                "pathogen_type": "Fungal",
                "severity": "High"
            },
            "Tomato Mosaic Virus": {
                "cause": "Tomato Mosaic Virus (viral pathogen)",
                "cure": "No chemical cure available - remove and destroy infected plants",
                "prevention": "Use certified virus-free seed, sanitise tools between plants, control insect vectors",
                "pathogen_type": "Viral",
                "severity": "High"
            },
            "Potato Late Blight": {
                "cause": "Phytophthora infestans (oomycete pathogen)",
                "cure": "Metalaxyl-based fungicides, apply every 5-7 days starting at disease onset",
                "prevention": "Adequate plant spacing, avoid overhead irrigation, remove infected tubers",
                "pathogen_type": "Fungal",
                "severity": "Critical"
            },
            "Potato Early Blight": {
                "cause": "Alternaria solani (fungal pathogen)",
                "cure": "Mancozeb, Chlorothalonil fungicides applied every 7-10 days",
                "prevention": "Destroy cull piles, control weeds, improve air circulation",
                "pathogen_type": "Fungal",
                "severity": "High"
            },
            "Cherry Powdery Mildew": {
                "cause": "Podosphaera clandestina (fungal pathogen)",
                "cure": "Sulfur or sterol-inhibitor fungicides applied weekly",
                "prevention": "Ensure adequate humidity is NOT present, improve sunlight penetration",
                "pathogen_type": "Fungal",
                "severity": "Medium"
            },
            "Grape Black Rot": {
                "cause": "Guignardia bidwellii (fungal pathogen)",
                "cure": "Mancozeb or Azoxystrobin applied every 10-14 days",
                "prevention": "Prune for airflow, remove mummified fruit clusters, destroy diseased canes",
                "pathogen_type": "Fungal",
                "severity": "High"
            },
            "Corn Leaf Blight": {
                "cause": "Helminthosporium species (fungal pathogen)",
                "cure": "Azoxystrobin or Propiconazole fungicides applied at heading",
                "prevention": "Use resistant varieties, crop rotation, destroy crop residue",
                "pathogen_type": "Fungal",
                "severity": "Medium"
            },
        }

model_manager = ModelManager()

# ============================================
# LIFESPAN CONTEXT MANAGER
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown"""
    # Startup
    logger.info("🚀 AgriSurgeon API starting...")
    model_manager.load_models()
    yield
    # Shutdown
    logger.info("⏹️ AgriSurgeon API shutting down...")

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title=Config.API_TITLE,
    description="AI-Powered Plant Disease Detection with IoT Integration",
    version=Config.API_VERSION,
    lifespan=lifespan
)

# ============================================
# CORS MIDDLEWARE
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# STATIC FILES
# ============================================
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static files directory not found")

# ============================================
# UTILITY FUNCTIONS
# ============================================
def process_image(file_content: bytes) -> np.ndarray:
    """Process uploaded image"""
    try:
        image = Image.open(io.BytesIO(file_content)).convert('RGB')
        image = image.resize((Config.IMG_SIZE, Config.IMG_SIZE))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def validate_image(file_content: bytes) -> bool:
    """Validate image file"""
    try:
        if len(file_content) > Config.MAX_FILE_SIZE:
            return False
        Image.open(io.BytesIO(file_content))
        return True
    except:
        return False

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🌱 AgriSurgeon API",
        "version": Config.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": Config.API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": model_manager.is_loaded
    }

@app.post("/api/predict")
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
        
        # Get advisory
        advisory = model_manager.advisory_db.get(
            disease_name,
            {
                "cause": "Unknown pathogen",
                "cure": "Consult agricultural specialist",
                "prevention": "Implement standard disease management practices",
                "pathogen_type": "Unknown",
                "severity": "Unknown"
            }
        )
        
        # Calculate risk level
        risk_level = calculate_risk_level(disease_name, temperature, humidity)
        
        logger.info(f"Prediction: {disease_name} (confidence: {confidence:.2%})")
        
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
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/api/batch-predict")
async def batch_predict(
    files: list[UploadFile] = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...)
):
    """Batch predict multiple images"""
    try:
        results = []
        for file in files:
            image_data = await file.read()
            if validate_image(image_data):
                image_array = process_image(image_data)
                visual_features = model_manager.feature_extractor.predict(
                    image_array,
                    verbose=0
                )[0]
                
                env_data = np.array([[temperature, humidity]])
                env_normalized = model_manager.scaler.transform(env_data)[0]
                
                hybrid_features = np.concatenate([visual_features, env_normalized])
                hybrid_features = hybrid_features.reshape(1, -1)
                
                disease_idx = model_manager.svm_model.predict(hybrid_features)[0]
                disease_name = model_manager.disease_names.get(disease_idx, "Unknown")
                
                decision_scores = model_manager.svm_model.decision_function(hybrid_features)[0]
                confidence = float((np.max(decision_scores) + 1) / 2)
                confidence = min(max(confidence, 0.0), 1.0)
                
                results.append({
                    "filename": file.filename,
                    "disease": disease_name,
                    "confidence": round(confidence, 4)
                })
        
        return {
            "total": len(files),
            "processed": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diseases")
async def get_diseases():
    """Get list of all detectable diseases"""
    return {
        "total": len(model_manager.disease_names),
        "diseases": list(model_manager.disease_names.values()),
        "advisory_available": len(model_manager.advisory_db)
    }

@app.get("/api/models")
async def get_models():
    """Get model information"""
    return {
        "models": [
            {
                "name": "MobileNetV3-Large",
                "parameters": 5400000,
                "inference_time_ms": 85
            },
            {
                "name": "Linear SVM",
                "classifiers": 38,
                "kernel": "linear"
            }
        ],
        "hybrid_features": 1282,
        "accuracy": 0.9850,
        "training_data": 87885
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get API statistics"""
    return {
        "version": Config.API_VERSION,
        "models_loaded": model_manager.is_loaded,
        "diseases_supported": len(model_manager.disease_names),
        "advisory_entries": len(model_manager.advisory_db),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================
# UTILITY FUNCTIONS
# ============================================
def calculate_risk_level(disease_name: str, temperature: float, humidity: float) -> str:
    """Calculate disease risk based on environmental factors"""
    
    risk_factors = {
        "Tomato Late Blight": {"min_temp": 10, "max_temp": 25, "min_humidity": 85},
        "Potato Late Blight": {"min_temp": 10, "max_temp": 25, "min_humidity": 85},
        "Apple Scab": {"min_temp": 5, "max_temp": 20, "min_humidity": 80},
        "Tomato Mosaic Virus": {"min_temp": 15, "max_temp": 30, "min_humidity": 40},
        "Cherry Powdery Mildew": {"min_temp": 15, "max_temp": 25, "min_humidity": 30},
    }
    
    factor = risk_factors.get(disease_name)
    if not factor:
        return "unknown"
    
    temp_ok = factor["min_temp"] <= temperature <= factor["max_temp"]
    humidity_ok = humidity >= factor["min_humidity"]
    
    if temp_ok and humidity_ok:
        return "high"
    elif temp_ok or humidity_ok:
        return "medium"
    else:
        return "low"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

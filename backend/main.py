"""
AgriSurgeon API - AI Powered Plant Disease Detection
Hybrid MobileNetV3 + SVM + IoT Environmental Fusion

FINAL VERSION WITH:
✔ Temperature
✔ Humidity
✔ Soil Moisture
✔ Environmental Disease Logic
✔ Dynamic Confidence Adjustment
✔ Risk Analysis
✔ Stable Feature Extraction
✔ Frontend Compatibility Fix
"""

import io
import json
import logging
from typing import List
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np

from tensorflow.keras.models import load_model

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from PIL import Image

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# CONFIGURATION
# =========================================================

class Config:

    API_TITLE = "AgriSurgeon API"

    API_DESCRIPTION = (
        "AI Powered Plant Disease Detection"
    )

    API_VERSION = "2.0.0"

    LOG_LEVEL = logging.INFO

    # =====================================================
    # PATHS
    # =====================================================

    MODEL_DIR = BASE_DIR / "models"

    MODEL_PATH = (
        MODEL_DIR / "MobileNetV3_SOTA_Regularized.h5"
    )

    SVM_PATH = (
        MODEL_DIR / "svm_hybrid_model.pkl"
    )

    SCALER_PATH = (
        MODEL_DIR / "scaler.pkl"
    )

    CLASS_INDICES_PATH = (
        MODEL_DIR / "class_indices.json"
    )

    DEPLOYMENT_CONFIG_PATH = (
        BASE_DIR / "deployment_config.json"
    )
    HISTORY_FILE = BASE_DIR / "prediction_history.json"

    # =====================================================
    # IMAGE
    # =====================================================

    IMG_SIZE = 224

    MAX_FILE_SIZE = 10 * 1024 * 1024

    ALLOWED_FORMATS = {
        "jpg",
        "jpeg",
        "png"
    }

    # =====================================================
    # CORS
    # =====================================================

    CORS_ORIGINS = ["*"]

    CORS_CREDENTIALS = True

    CORS_METHODS = ["*"]

    CORS_HEADERS = ["*"]

    # =====================================================
    # MODEL INFO
    # =====================================================

    MODEL_ACCURACY = 0.985

    DISEASE_CATEGORIES = 38

    PLANT_SPECIES = 14

    TRAINING_SAMPLES = 54000


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=(
        "%(asctime)s - "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)

# =========================================================
# MODEL MANAGER
# =========================================================

class ModelManager:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super(
                ModelManager,
                cls
            ).__new__(cls)

            cls._instance.is_loaded = False

        return cls._instance

    # =====================================================
    # LOAD MODELS
    # =====================================================

    def load_models(self):

        try:

            logger.info(
                "🔄 Loading trained models..."
            )

            required_files = [

                Config.MODEL_PATH,

                Config.SVM_PATH,

                Config.SCALER_PATH,

                Config.CLASS_INDICES_PATH
            ]

            for file_path in required_files:

                logger.info(
                    f"Checking: {file_path}"
                )

                if not file_path.exists():

                    raise FileNotFoundError(
                        f"Missing file: {file_path}"
                    )

            # =================================================
            # LOAD CNN MODEL
            # =================================================

            logger.info(
                "📥 Loading CNN model..."
            )

            self.cnn_model = load_model(
                str(Config.MODEL_PATH)
            )

            logger.info(
                "✓ CNN model loaded"
            )

            # =================================================
            # FEATURE EXTRACTOR
            # =================================================

            logger.info(
                "📥 Creating feature extractor..."
            )

            self.feature_extractor = (
                self.cnn_model.layers[0]
            )

            logger.info(
                f"✓ Feature extractor ready | "
                f"Output Shape: "
                f"{self.feature_extractor.output_shape}"
            )

            # =================================================
            # LOAD SCALER
            # =================================================

            logger.info(
                "📥 Loading scaler..."
            )

            self.scaler = joblib.load(
                str(Config.SCALER_PATH)
            )

            logger.info(
                "✓ Scaler loaded"
            )

            logger.info(
                f"Scaler expects "
                f"{self.scaler.n_features_in_} "
                f"features"
            )

            # =================================================
            # LOAD SVM
            # =================================================

            logger.info(
                "📥 Loading SVM..."
            )

            self.svm_model = joblib.load(
                str(Config.SVM_PATH)
            )

            logger.info(
                "✓ SVM loaded"
            )

            # =================================================
            # LOAD CLASS INDICES
            # =================================================

            logger.info(
                "📥 Loading class indices..."
            )

            with open(
                str(Config.CLASS_INDICES_PATH),
                "r"
            ) as f:

                class_indices = json.load(f)

            self.disease_names = {
                v: k
                for k, v in class_indices.items()
            }

            logger.info(
                f"✓ Loaded "
                f"{len(self.disease_names)} classes"
            )

            # =================================================
            # ADVISORY DATABASE
            # =================================================

            if Config.DEPLOYMENT_CONFIG_PATH.exists():

                with open(
                    str(
                        Config.DEPLOYMENT_CONFIG_PATH
                    ),
                    "r"
                ) as f:

                    config_data = json.load(f)

                    self.advisory_db = (
                        config_data.get(
                            "advisory",
                            {}
                        )
                    )

            else:

                self.advisory_db = {}

            logger.info(
                "✅ ALL MODELS LOADED SUCCESSFULLY"
            )

            self.is_loaded = True

        except Exception as e:

            logger.error(
                f"❌ Error loading models: {e}",
                exc_info=True
            )

            raise


# =========================================================
# INITIALIZE MODEL MANAGER
# =========================================================

model_manager = ModelManager()

# =========================================================
# FASTAPI LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    model_manager.load_models()

    yield

    logger.info(
        "⏹️ Shutting down API"
    )

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    lifespan=lifespan
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=Config.CORS_CREDENTIALS,
    allow_methods=Config.CORS_METHODS,
    allow_headers=Config.CORS_HEADERS,
)

# =========================================================
# RESPONSE MODEL
# =========================================================

class PredictionResponse(BaseModel):

    class Config:
        protected_namespaces = ()

    disease: str

    confidence: float

    temperature: float

    humidity: float

    soil_moisture: float

    environmental_risk: str

    environmental_factor: str

    advisory: dict

    timestamp: str

    model_version: str


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image(image_data: bytes):

    try:

        if (
            len(image_data)
            > Config.MAX_FILE_SIZE
        ):

            return False

        img = Image.open(
            io.BytesIO(image_data)
        )

        img.verify()

        img = Image.open(
            io.BytesIO(image_data)
        )

        if (
            img.format.lower()
            not in Config.ALLOWED_FORMATS
        ):

            return False

        return True

    except Exception as e:

        logger.error(
            f"Image validation error: {e}"
        )

        return False


# =========================================================
# PROCESS IMAGE
# =========================================================

def process_image(image_data: bytes):

    img = Image.open(
        io.BytesIO(image_data)
    ).convert("RGB")

    img = img.resize(
        (
            Config.IMG_SIZE,
            Config.IMG_SIZE
        )
    )

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array.astype(np.float32)


# =========================================================
# ENVIRONMENTAL ANALYSIS
# =========================================================

def calculate_environmental_factor(
    disease_name,
    temperature,
    humidity,
    soil_moisture
):

    score = 0

    reasons = []

    # =====================================================
    # HUMIDITY
    # =====================================================

    if humidity > 80:

        score += 1

        reasons.append(
            "High humidity supports fungal growth"
        )

    # =====================================================
    # SOIL MOISTURE
    # =====================================================

    if soil_moisture > 75:

        score += 1

        reasons.append(
            "Excess soil moisture increases infection risk"
        )

    # =====================================================
    # TEMPERATURE
    # =====================================================

    if temperature > 32:

        score += 1

        reasons.append(
            "High temperature stresses plants"
        )

    # =====================================================
    # DISEASE SPECIFIC
    # =====================================================

    if "blight" in disease_name.lower():

        if humidity > 75:

            score += 1

            reasons.append(
                "Blight diseases spread rapidly in humidity"
            )

    if "mold" in disease_name.lower():

        if humidity > 70:

            score += 1

            reasons.append(
                "Mold growth favored by humid climate"
            )

    # =====================================================
    # FINAL LEVEL
    # =====================================================

    if score >= 3:

        level = "High Environmental Influence"

        multiplier = 1.25

    elif score == 2:

        level = "Moderate Environmental Influence"

        multiplier = 1.12

    elif score == 1:

        level = "Low Environmental Influence"

        multiplier = 1.05

    else:

        level = "Minimal Environmental Influence"

        multiplier = 0.95

    return {

        "level": level,

        "multiplier": multiplier,

        "reasons": reasons
    }


# =========================================================
# RISK LEVEL
# =========================================================

def calculate_risk_level(
    disease_name,
    temperature,
    humidity,
    soil_moisture
):

    if "healthy" in disease_name.lower():

        return "none"

    risk_score = 0

    if humidity > 75:
        risk_score += 1

    if soil_moisture > 70:
        risk_score += 1

    if temperature > 32:
        risk_score += 1

    if risk_score >= 3:

        return "high"

    elif risk_score == 2:

        return "medium"

    return "low"
# =========================================================
# HISTORY FUNCTIONS
# =========================================================

def load_prediction_history():

    try:

        if not Config.HISTORY_FILE.exists():

            return []

        with open(
            Config.HISTORY_FILE,
            "r"
        ) as f:

            return json.load(f)

    except Exception as e:

        logger.error(
            f"History load error: {e}"
        )

        return []


def save_prediction_history(entry):

    try:

        history = load_prediction_history()

        history.insert(0, entry)

        # KEEP LAST 100 RECORDS
        history = history[:100]

        with open(
            Config.HISTORY_FILE,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )

    except Exception as e:

        logger.error(
            f"History save error: {e}"
        )

# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():
    history_entry = {

        "disease": disease_name,

        "confidence": round(
            final_confidence,
            4
        ),

        "temperature": temperature,

        "humidity": humidity,

        "soil_moisture": soil_moisture,

        "environmental_risk": risk_level,

        "environmental_factor": (
            env_analysis["level"]
        ),

        "timestamp": (
            datetime.utcnow().isoformat()
        )
    }

    save_prediction_history(
        history_entry
    )

    return {

        "name": Config.API_TITLE,

        "version": Config.API_VERSION,

        "status": "running",

        "models_loaded": model_manager.is_loaded
        
        # =================================================
        # SAVE HISTORY
        # =================================================

       
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
async def health_check():

    return {

        "status": "healthy",

        "models_loaded": (
            model_manager.is_loaded
        ),

        "timestamp": (
            datetime.utcnow().isoformat()
        )
    }


# =========================================================
# PREDICT
# =========================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse
)
async def predict(

    file: UploadFile = File(...),

    temperature: float = Form(...),

    humidity: float = Form(...),

    # SUPPORT BOTH FIELD NAMES
    soil_moisture: float = Form(None),

    soilMoisture: float = Form(None)

):

    try:

        # =================================================
        # MODEL CHECK
        # =================================================

        if not model_manager.is_loaded:

            raise HTTPException(
                status_code=503,
                detail="Models not loaded"
            )

        # =================================================
        # SUPPORT BOTH FRONTEND FIELD NAMES
        # =================================================

        if soil_moisture is None:

            soil_moisture = soilMoisture

        # =================================================
        # DEFAULT VALUE
        # =================================================

        if soil_moisture is None:

            soil_moisture = 50.0

        # =================================================
        # READ IMAGE
        # =================================================

        image_data = await file.read()

        if not validate_image(image_data):

            raise HTTPException(
                status_code=400,
                detail="Invalid image"
            )

        image_array = process_image(
            image_data
        )

        # =================================================
        # FEATURE EXTRACTION
        # =================================================

        visual_features = (
            model_manager.feature_extractor.predict(
                image_array,
                verbose=0
            )[0]
        )

        logger.info(
            f"Visual Features Shape: "
            f"{visual_features.shape}"
        )

        # =================================================
        # ENVIRONMENTAL FEATURES
        # =================================================

        env_features = np.array([

            temperature,

            humidity

        ])

        # =================================================
        # HYBRID FEATURES
        # =================================================

        hybrid_features = np.concatenate([

            visual_features,

            env_features

        ])

        hybrid_features = (
            hybrid_features.reshape(1, -1)
        )

        logger.info(
            f"Hybrid Features Shape: "
            f"{hybrid_features.shape}"
        )

        # =================================================
        # FEATURE CHECK
        # =================================================

        expected_features = (
            model_manager.scaler.n_features_in_
        )

        current_features = (
            hybrid_features.shape[1]
        )

        if current_features != expected_features:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Feature mismatch | "
                    f"Expected "
                    f"{expected_features} "
                    f"but got "
                    f"{current_features}"
                )
            )

        # =================================================
        # SCALE
        # =================================================

        hybrid_features = (
            model_manager.scaler.transform(
                hybrid_features
            )
        )

        # =================================================
        # SVM PREDICTION
        # =================================================

        disease_idx = (
            model_manager.svm_model.predict(
                hybrid_features
            )[0]
        )

        disease_name = (
            model_manager.disease_names.get(
                disease_idx,
                "Unknown Disease"
            )
        )

        # =================================================
        # CONFIDENCE
        # =================================================

        decision_scores = (
            model_manager.svm_model.decision_function(
                hybrid_features
            )[0]
        )

        base_confidence = float(
            (np.max(decision_scores) + 1) / 2
        )

        base_confidence = min(
            max(base_confidence, 0.0),
            1.0
        )

        # =================================================
        # ENVIRONMENTAL ANALYSIS
        # =================================================

        env_analysis = (
            calculate_environmental_factor(
                disease_name,
                temperature,
                humidity,
                soil_moisture
            )
        )

        final_confidence = (
            base_confidence
            * env_analysis["multiplier"]
        )

        final_confidence = min(
            final_confidence,
            1.0
        )

        # =================================================
        # RISK
        # =================================================

        risk_level = calculate_risk_level(
            disease_name,
            temperature,
            humidity,
            soil_moisture
        )

        # =================================================
        # ADVISORY
        # =================================================

        advisory = (
            model_manager.advisory_db.get(
                disease_name,
                {
                    "cause": (
                        "Environmental conditions "
                        "may contribute to disease spread"
                    ),

                    "cure": (
                        "Consult agricultural expert"
                    ),

                    "prevention": (
                        "Maintain balanced moisture "
                        "and humidity"
                    ),

                    "severity": "Unknown"
                }
            )
        )

        # =================================================
        # LOGGING
        # =================================================

        logger.info(
            f"Prediction: {disease_name}"
        )

        logger.info(
            f"Base Confidence: "
            f"{base_confidence:.2%}"
        )

        logger.info(
            f"Final Confidence: "
            f"{final_confidence:.2%}"
        )

        logger.info(
            f"Environmental Influence: "
            f"{env_analysis['level']}"
        )

        # =================================================
        # RESPONSE
        # =================================================

        return {

            "disease": disease_name,

            "confidence": round(
                final_confidence,
                4
            ),

            "temperature": temperature,

            "humidity": humidity,

            "soil_moisture": soil_moisture,

            "environmental_risk": risk_level,

            "environmental_factor": (
                env_analysis["level"]
            ),

            "advisory": advisory,

            "timestamp": (
                datetime.utcnow().isoformat()
            ),

            "model_version": (
                Config.API_VERSION
            )
        }

    except HTTPException as e:

        raise e

    except Exception as e:

        logger.error(
            f"Prediction error: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# DISEASES
# =========================================================

@app.get("/api/diseases")
async def get_diseases():

    return {

        "count": len(
            model_manager.disease_names
        ),

        "diseases": sorted(
            list(
                model_manager.disease_names.values()
            )
        )
    }


# =========================================================
# MODEL INFO
# =========================================================

@app.get("/api/models")
async def get_models():

    return {

        "cnn": "MobileNetV3",

        "svm": "Hybrid SVM",

        "accuracy": Config.MODEL_ACCURACY,

        "classes": (
            Config.DISEASE_CATEGORIES
        )
    }
    # =========================================================
# PREDICTION HISTORY
# =========================================================

@app.get("/api/history")
async def get_history():

    history = load_prediction_history()

    return {

        "count": len(history),

        "history": history
    }


# =========================================================
# CLEAR HISTORY
# =========================================================

@app.delete("/api/history")
async def clear_history():

    try:

        with open(
            Config.HISTORY_FILE,
            "w"
        ) as f:

            json.dump([], f)

        return {

            "message": "History cleared"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# ERROR HANDLER
# =========================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request,
    exc
):

    return JSONResponse(

        status_code=exc.status_code,

        content={

            "error": exc.detail,

            "status_code": exc.status_code,

            "timestamp": (
                datetime.utcnow().isoformat()
            )
        }
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    logger.info(
        "🚀 Starting AgriSurgeon API"
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

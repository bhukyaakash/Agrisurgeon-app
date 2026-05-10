"""
Configuration Management for AgriSurgeon API
"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # API Configuration
    API_TITLE = "AgriSurgeon API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = "Hybrid MobileNetV3-SVM-IoT Fusion Framework for Intelligent Plant Disease Detection"
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    
    # Model paths
    MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "MobileNetV3_SOTA_Regularized.h5"))
    SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(MODELS_DIR, "scaler.pkl"))
    SVM_PATH = os.getenv("SVM_PATH", os.path.join(MODELS_DIR, "svm_hybrid_model.pkl"))
    CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH", os.path.join(MODELS_DIR, "class_indices.json"))
    DEPLOYMENT_CONFIG_PATH = BASE_DIR / "deployment_config.json"
    
    # Image processing
    IMG_SIZE = 224
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FORMATS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    
    # Model performance metrics
    MODEL_ACCURACY = 0.9850
    MODEL_PRECISION = 0.9851
    MODEL_RECALL = 0.9850
    MODEL_F1_SCORE = 0.9850
    
    # Environmental sensor ranges
    TEMP_MIN = 10
    TEMP_MAX = 50
    HUMIDITY_MIN = 0
    HUMIDITY_MAX = 100
    SOIL_MOISTURE_MIN = 0
    SOIL_MOISTURE_MAX = 100
    
    # Disease categories
    DISEASE_CATEGORIES = 38
    PLANT_SPECIES = 14
    TRAINING_SAMPLES = 87885
    
    # Feature dimensions
    VISUAL_FEATURES_DIM = 1280
    ENVIRONMENTAL_FEATURES_DIM = 2
    HYBRID_FEATURES_DIM = 1282
    
    # Inference configuration
    BATCH_SIZE = 32
    INFERENCE_TIMEOUT = 30  # seconds
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
    
    # Database (future)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agrisurgeon.db")
    
    # CORS
    CORS_ORIGINS = ["*"]
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    @classmethod
    def validate_environment_data(cls, temperature: float, humidity: float, soil_moisture: float) -> bool:
        """Validate environmental data"""
        return (
            cls.TEMP_MIN <= temperature <= cls.TEMP_MAX and
            cls.HUMIDITY_MIN <= humidity <= cls.HUMIDITY_MAX and
            cls.SOIL_MOISTURE_MIN <= soil_moisture <= cls.SOIL_MOISTURE_MAX
        )

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = "sqlite:///./test.db"

def get_config() -> Config:
    """Get appropriate configuration based on environment"""
    env = os.getenv("ENV", "development")
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    return DevelopmentConfig()

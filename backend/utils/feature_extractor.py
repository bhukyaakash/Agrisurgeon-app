"""
Feature Extraction and Fusion Utilities
"""

import logging
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extraction and hybrid fusion"""
    
    VISUAL_FEATURES_DIM = 1280
    ENVIRONMENTAL_FEATURES_DIM = 2
    HYBRID_FEATURES_DIM = 1282
    
    @staticmethod
    def validate_visual_features(features: np.ndarray) -> Tuple[bool, str]:
        """Validate visual features from CNN"""
        try:
            if len(features.shape) != 1:
                return False, f"Expected 1D array, got shape {features.shape}"
            
            if len(features) != FeatureExtractor.VISUAL_FEATURES_DIM:
                return False, f"Expected {FeatureExtractor.VISUAL_FEATURES_DIM} features, got {len(features)}"
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return False, "Features contain NaN or Inf values"
            
            return True, "Valid"
        
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate_environmental_data(temperature: float, humidity: float) -> Tuple[bool, str]:
        """Validate environmental measurements"""
        try:
            if not isinstance(temperature, (int, float)):
                return False, "Temperature must be numeric"
            
            if not isinstance(humidity, (int, float)):
                return False, "Humidity must be numeric"
            
            if temperature < -50 or temperature > 60:
                return False, "Temperature out of reasonable range (-50 to 60°C)"
            
            if humidity < 0 or humidity > 100:
                return False, "Humidity must be between 0-100%"
            
            return True, "Valid"
        
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def normalize_environmental_data(
        temperature: float,
        humidity: float,
        scaler: StandardScaler = None
    ) -> Tuple[np.ndarray, str]:
        """Normalize environmental data using fitted scaler"""
        try:
            env_data = np.array([[temperature, humidity]])
            
            if scaler is None:
                logger.warning("No scaler provided, using raw values")
                return env_data[0], "Warning: No normalization"
            
            normalized = scaler.transform(env_data)
            
            return normalized[0], "Success"
        
        except Exception as e:
            logger.error(f"Normalization error: {str(e)}")
            return None, str(e)
    
    @staticmethod
    def create_hybrid_features(
        visual_features: np.ndarray,
        temperature: float,
        humidity: float,
        scaler: StandardScaler = None
    ) -> Tuple[np.ndarray, str]:
        """Create hybrid feature vector from visual and environmental data"""
        try:
            # Validate visual features
            valid, msg = FeatureExtractor.validate_visual_features(visual_features)
            if not valid:
                return None, f"Invalid visual features: {msg}"
            
            # Validate environmental data
            valid, msg = FeatureExtractor.validate_environmental_data(temperature, humidity)
            if not valid:
                return None, f"Invalid environmental data: {msg}"
            
            # Normalize environmental data
            env_normalized, msg = FeatureExtractor.normalize_environmental_data(
                temperature, humidity, scaler
            )
            
            if env_normalized is None:
                logger.warning("Using raw environmental data due to normalization error")
                env_normalized = np.array([temperature, humidity])
            
            # Concatenate features
            hybrid_features = np.concatenate([visual_features, env_normalized])
            
            # Validate hybrid features
            if len(hybrid_features) != FeatureExtractor.HYBRID_FEATURES_DIM:
                return None, f"Hybrid feature dimension mismatch: {len(hybrid_features)}"
            
            logger.debug(f"Hybrid features created: shape={hybrid_features.shape}")
            
            return hybrid_features, "Success"
        
        except Exception as e:
            logger.error(f"Hybrid feature creation error: {str(e)}")
            return None, str(e)
    
    @staticmethod
    def calculate_feature_importance(svm_weights: np.ndarray) -> dict:
        """Calculate feature importance from SVM weights"""
        try:
            importance = {
                "visual_importance": float(np.mean(np.abs(svm_weights[:1280]))),
                "environmental_importance": float(np.mean(np.abs(svm_weights[1280:]))),
                "total_dimensions": len(svm_weights),
                "visual_dimensions": 1280,
                "environmental_dimensions": 2
            }
            
            return importance
        
        except Exception as e:
            logger.error(f"Feature importance calculation error: {str(e)}")
            return {}
    
    @staticmethod
    def create_feature_matrix(
        visual_features_list: list,
        environmental_data_list: list,
        scaler: StandardScaler = None
    ) -> Tuple[np.ndarray, str]:
        """Create feature matrix from multiple samples"""
        try:
            if len(visual_features_list) != len(environmental_data_list):
                return None, "Mismatched number of samples"
            
            hybrid_features_list = []
            
            for visual_features, (temp, humidity) in zip(
                visual_features_list, environmental_data_list
            ):
                hybrid, msg = FeatureExtractor.create_hybrid_features(
                    visual_features, temp, humidity, scaler
                )
                
                if hybrid is None:
                    return None, msg
                
                hybrid_features_list.append(hybrid)
            
            feature_matrix = np.array(hybrid_features_list)
            
            logger.info(f"Feature matrix created: shape={feature_matrix.shape}")
            
            return feature_matrix, "Success"
        
        except Exception as e:
            logger.error(f"Feature matrix creation error: {str(e)}")
            return None, str(e)

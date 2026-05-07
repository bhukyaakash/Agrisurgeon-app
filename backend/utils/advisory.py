"""
Agricultural Advisory System for Disease Management
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class AgronomicAdvisory:
    """Provide agricultural advisory based on disease prediction"""
    
    DISEASE_DATABASE = {
        "Apple Scab": {
            "cause": "Venturia inaequalis (fungal pathogen)",
            "symptoms": "Olive-brown lesions on leaves, fruits, and twigs; cause russetting on fruit",
            "cure": "Captan or Mancozeb fungicide applications at 7-14 day intervals during growing season",
            "active_ingredients": ["Captan", "Mancozeb", "Sulfur"],
            "prevention": "Remove fallen leaves; improve air circulation; prune for better canopy airflow; avoid overhead irrigation",
            "pathogen_type": "Fungal",
            "severity": "High",
            "environmental_preference": "Cool, wet conditions (15-25°C, >85% RH)",
            "quarantine_required": False
        },
        "Apple Rust": {
            "cause": "Gymnosporangium species (fungal pathogen)",
            "symptoms": "Orange, cushion-like growths on leaf undersides; yellow spots on upper surface",
            "cure": "Myclobutanil or Triadimefon fungicides applied at bud break and 2-3 week intervals",
            "active_ingredients": ["Myclobutanil", "Triadimefon"],
            "prevention": "Remove alternate host plants (junipers); improve drainage; remove infected twigs",
            "pathogen_type": "Fungal",
            "severity": "Medium",
            "environmental_preference": "Moderate humidity with alternate wet-dry cycles",
            "quarantine_required": False
        },
        "Tomato Late Blight": {
            "cause": "Phytophthora infestans (oomycete pathogen)",
            "symptoms": "Water-soaked lesions on leaves and stems; white mold on leaf undersides; fruit rot",
            "cure": "Metalaxyl (Ridomil) or Dimethomorph systemic fungicides applied every 5-7 days at disease onset",
            "active_ingredients": ["Metalaxyl", "Dimethomorph", "Mancozeb"],
            "prevention": "Avoid overhead irrigation; reduce canopy humidity to <85%; remove infected leaves; ensure 90cm plant spacing",
            "pathogen_type": "Fungal (Oomycete)",
            "severity": "Critical",
            "environmental_preference": "Cool (15-20°C), wet conditions (>90% RH) - CRITICAL THRESHOLD",
            "quarantine_required": True
        },
        "Tomato Early Blight": {
            "cause": "Alternaria solani (fungal pathogen)",
            "symptoms": "Concentric ring lesions on leaves; target-like spots; premature defoliation",
            "cure": "Chlorothalonil or Mancozeb fungicides applied preventatively every 7-10 days",
            "active_ingredients": ["Chlorothalonil", "Mancozeb", "Copper"],
            "prevention": "Remove lower leaves; improve air circulation; mulch soil to prevent splash; destroy fallen leaves",
            "pathogen_type": "Fungal",
            "severity": "High",
            "environmental_preference": "Moderate temperature (25-30°C), low-moderate humidity (60-80%)",
            "quarantine_required": False
        },
        "Tomato Mosaic Virus": {
            "cause": "Tomato Mosaic Virus (ToMV) - viral pathogen",
            "symptoms": "Mottled leaf pattern; curling leaves; stunted growth; reduced fruit set",
            "cure": "NO CHEMICAL CURE - Remove and destroy infected plants immediately",
            "active_ingredients": [],
            "prevention": "Use certified virus-free seed; sanitise tools between plants with 10% bleach; control insect vectors; avoid tobacco use near plants",
            "pathogen_type": "Viral",
            "severity": "High",
            "environmental_preference": "Wide temperature range; spread mainly through mechanical contact",
            "quarantine_required": True
        },
        "Potato Late Blight": {
            "cause": "Phytophthora infestans (oomycete pathogen)",
            "symptoms": "Water-soaked lesions on leaves and stems; white mold on undersides; dark, mushy tuber rot",
            "cure": "Metalaxyl or Dimethomorph fungicides applied every 5-7 days starting at disease onset",
            "active_ingredients": ["Metalaxyl", "Dimethomorph", "Mancozeb"],
            "prevention": "Adequate plant spacing (60cm); avoid overhead irrigation; remove infected tubers; use certified seed potatoes",
            "pathogen_type": "Fungal (Oomycete)",
            "severity": "Critical",
            "environmental_preference": "Cool (10-20°C), wet conditions (>90% RH) - CRITICAL THRESHOLD",
            "quarantine_required": True
        },
        "Potato Early Blight": {
            "cause": "Alternaria solani (fungal pathogen)",
            "symptoms": "Concentric ring spots on lower leaves; target-like lesions; premature defoliation",
            "cure": "Mancozeb or Chlorothalonil fungicides applied every 7-10 days",
            "active_ingredients": ["Mancozeb", "Chlorothalonil"],
            "prevention": "Destroy cull piles; control weeds; improve air circulation; remove infected leaves",
            "pathogen_type": "Fungal",
            "severity": "High",
            "environmental_preference": "Moderate temperature (20-28°C), moderate humidity (70-85%)",
            "quarantine_required": False
        },
        "Cherry Powdery Mildew": {
            "cause": "Podosphaera clandestina (fungal pathogen)",
            "symptoms": "White powdery coating on leaves and young shoots; leaf curling; distorted growth",
            "cure": "Sulfur or sterol-inhibitor (DMI) fungicides applied weekly when conditions favor disease",
            "active_ingredients": ["Sulfur", "Triazoles", "Pyrimethamine"],
            "prevention": "Ensure adequate air circulation; prune to open canopy; avoid excessive nitrogen fertilizer",
            "pathogen_type": "Fungal",
            "severity": "Medium",
            "environmental_preference": "High temperature (18-25°C), LOW humidity (<50%) - develops best in dry conditions",
            "quarantine_required": False
        },
        "Grape Black Rot": {
            "cause": "Guignardia bidwellii (fungal pathogen)",
            "symptoms": "Brown lesions with dark centers on leaves; rotten berries with black, hardened appearance",
            "cure": "Mancozeb or Azoxystrobin fungicides applied every 10-14 days during growing season",
            "active_ingredients": ["Mancozeb", "Azoxystrobin", "Captan"],
            "prevention": "Prune for good air circulation; remove mummified fruit clusters; destroy diseased canes; improve canopy airflow",
            "pathogen_type": "Fungal",
            "severity": "High",
            "environmental_preference": "Warm (20-28°C), humid conditions (75-90% RH)",
            "quarantine_required": False
        },
        "Corn Leaf Blight": {
            "cause": "Helminthosporium species or Exserohilum (fungal pathogens)",
            "symptoms": "Rectangular lesions with tan centers and dark borders on leaves; premature plant death",
            "cure": "Azoxystrobin or Propiconazole fungicides applied at boot to heading stage",
            "active_ingredients": ["Azoxystrobin", "Propiconazole"],
            "prevention": "Use resistant varieties; practice 2-year crop rotation; destroy crop residue; avoid overhead irrigation",
            "pathogen_type": "Fungal",
            "severity": "Medium",
            "environmental_preference": "Cool to moderate temperature (15-25°C), moderate humidity (60-80%)",
            "quarantine_required": False
        }
    }
    
    @staticmethod
    def get_advisory(disease_name: str) -> Dict:
        """Get advisory for a specific disease"""
        try:
            if disease_name in AgronomicAdvisory.DISEASE_DATABASE:
                return AgronomicAdvisory.DISEASE_DATABASE[disease_name]
            else:
                logger.warning(f"Advisory not found for disease: {disease_name}")
                return AgronomicAdvisory._get_default_advisory(disease_name)
        
        except Exception as e:
            logger.error(f"Error retrieving advisory: {str(e)}")
            return {}
    
    @staticmethod
    def _get_default_advisory(disease_name: str) -> Dict:
        """Get default advisory for unknown diseases"""
        return {
            "cause": "Unknown pathogen",
            "symptoms": "Visual symptoms detected - consult agricultural specialist for accurate diagnosis",
            "cure": "Consult with local agricultural extension office or qualified plant pathologist",
            "active_ingredients": [],
            "prevention": "Implement general disease management practices: improve air circulation, reduce overhead irrigation, practice crop rotation",
            "pathogen_type": "Unknown",
            "severity": "Unknown",
            "environmental_preference": "Unknown",
            "quarantine_required": False
        }
    
    @staticmethod
    def get_contextual_advisory(
        disease_name: str,
        temperature: float,
        humidity: float
    ) -> Dict:
        """Get context-aware advisory based on environmental conditions"""
        try:
            advisory = AgronomicAdvisory.get_advisory(disease_name)
            
            # Add urgency assessment
            urgency = AgronomicAdvisory._assess_urgency(
                disease_name, temperature, humidity
            )
            advisory["urgency"] = urgency
            
            # Add environmental assessment
            env_assessment = AgronomicAdvisory._assess_environmental_conditions(
                disease_name, temperature, humidity
            )
            advisory["environmental_assessment"] = env_assessment
            
            return advisory
        
        except Exception as e:
            logger.error(f"Error generating contextual advisory: {str(e)}")
            return {}
    
    @staticmethod
    def _assess_urgency(disease_name: str, temperature: float, humidity: float) -> str:
        """Assess urgency of treatment"""
        critical_diseases = ["Tomato Late Blight", "Potato Late Blight", "Citrus Greening"]
        
        if disease_name in critical_diseases:
            return "CRITICAL - Immediate action required"
        
        # Check if conditions favor disease progression
        advisory = AgronomicAdvisory.get_advisory(disease_name)
        env_pref = advisory.get("environmental_preference", "")
        
        if "CRITICAL THRESHOLD" in env_pref:
            if temperature < 20 and humidity > 85:
                return "HIGH - Conditions highly favorable for disease progression"
        
        return "STANDARD - Monitor and apply treatment as scheduled"
    
    @staticmethod
    def _assess_environmental_conditions(
        disease_name: str,
        temperature: float,
        humidity: float
    ) -> Dict:
        """Assess whether environmental conditions favor disease"""
        conditions = {
            "temperature": temperature,
            "humidity": humidity,
            "conditions_favorable": False,
            "recommendation": ""
        }
        
        # Disease-specific environmental preferences
        disease_conditions = {
            "Tomato Late Blight": {"min_temp": 10, "max_temp": 25, "min_humidity": 85},
            "Potato Late Blight": {"min_temp": 10, "max_temp": 25, "min_humidity": 85},
            "Apple Scab": {"min_temp": 5, "max_temp": 20, "min_humidity": 80},
            "Cherry Powdery Mildew": {"min_temp": 15, "max_temp": 25, "min_humidity": 30},
        }
        
        if disease_name in disease_conditions:
            pref = disease_conditions[disease_name]
            temp_ok = pref["min_temp"] <= temperature <= pref["max_temp"]
            humidity_ok = humidity >= pref["min_humidity"]
            
            if temp_ok and humidity_ok:
                conditions["conditions_favorable"] = True
                conditions["recommendation"] = "Environmental conditions are HIGHLY FAVORABLE for disease development. Increase treatment frequency."
            elif temp_ok or humidity_ok:
                conditions["conditions_favorable"] = True
                conditions["recommendation"] = "Some environmental conditions favor disease. Monitor closely."
            else:
                conditions["conditions_favorable"] = False
                conditions["recommendation"] = "Current environmental conditions do NOT favor disease development."
        
        return conditions
    
    @staticmethod
    def get_all_diseases() -> List[str]:
        """Get list of all diseases with available advisory"""
        return list(AgronomicAdvisory.DISEASE_DATABASE.keys())
    
    @staticmethod
    def get_treatment_protocols(disease_name: str) -> Dict:
        """Get detailed treatment protocols"""
        advisory = AgronomicAdvisory.get_advisory(disease_name)
        
        return {
            "disease": disease_name,
            "active_ingredients": advisory.get("active_ingredients", []),
            "application_frequency": AgronomicAdvisory._get_application_frequency(disease_name),
            "safety_precautions": ["Use protective equipment", "Apply during cool hours", "Avoid applicatin before rain"],
            "cost_effectiveness": AgronomicAdvisory._assess_cost_effectiveness(disease_name)
        }
    
    @staticmethod
    def _get_application_frequency(disease_name: str) -> str:
        """Get fungicide application frequency"""
        frequency_map = {
            "Tomato Late Blight": "Every 5-7 days",
            "Potato Late Blight": "Every 5-7 days",
            "Apple Scab": "Every 7-14 days",
            "Tomato Early Blight": "Every 7-10 days",
            "Grape Black Rot": "Every 10-14 days",
        }
        return frequency_map.get(disease_name, "As recommended by extension office")
    
    @staticmethod
    def _assess_cost_effectiveness(disease_name: str) -> Dict:
        """Assess cost-benefit of treatment"""
        return {
            "estimated_yield_loss_untreated": "20-40%",
            "estimated_treatment_cost": "Low-Medium",
            "recommendation": "Treatment is cost-effective for major crops"
        }

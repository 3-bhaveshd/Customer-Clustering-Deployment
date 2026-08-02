import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    CATEGORIES: List[str] = field(default_factory=lambda: [
        "Steady Middle-Grounders",
        "High-Value Elites",
        "Young Spenders",
        "Conservative Wealthy",
        "Affluent Influencers",
        "Budget-Conscious Seniors"
    ])
    
    N_CLUSTERS: int = 6
    RANDOM_STATE: int = 42
    
    NUMERICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "Age", "Annual_Income", "Spending_Score"
    ])
    GENDER_FEATURE: str = "Gender"
    
    DATA_PATH: str = os.path.join(BASE_DIR, "Mall_Customers.csv")
    PREPROCESSOR_PATH: str = os.path.join(BASE_DIR, "preprocessor.joblib")
    MODEL_PATH: str = os.path.join(BASE_DIR, "kmeans_model.joblib")
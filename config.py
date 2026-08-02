import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    CATEGORIES: List[str] = field(default_factory=lambda: [
        "Nice, Properly Spending your money !",
        "High-Value Elites",
        "Sadly, You are Poor !",
        "Conservative Wealthy",
        "Blind Spending !",
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

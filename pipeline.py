import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from config import Config

class DataPreprocessor:
    """Handles data transformation, scaling, and categorical encoding."""
    
    def __init__(self, config: Config):
        self.config = config
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.config.NUMERICAL_FEATURES),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), [self.config.GENDER_FEATURE])
            ]
        )

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transformer.fit_transform(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(df)

    def save(self, path: str):
        joblib.dump(self.transformer, path)

    def load(self, path: str):
        self.transformer = joblib.load(path)


class CustomerClusteringModel:
    """Handles KMeans model training, prediction, and artifact loading."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = KMeans(
            n_clusters=self.config.N_CLUSTERS,
            random_state=self.config.RANDOM_STATE,
            n_init=10
        )

    def train(self, X: np.ndarray):
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)


class InferencePipeline:
    """Orchestrates loading saved artifacts and making real-time predictions."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.model_wrapper = CustomerClusteringModel(config)
        self._load_artifacts()

    def _load_artifacts(self):
        self.preprocessor.load(self.config.PREPROCESSOR_PATH)
        self.model_wrapper.load(self.config.MODEL_PATH)

    def predict_segment(self, gender: str, age: int, annual_income: float, spending_score: int) -> str:
        # Construct single row DataFrame matching original structure
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score
        }])
        
        # Transform input and predict cluster index
        transformed_data = self.preprocessor.transform(input_data)
        cluster_idx = self.model_wrapper.predict(transformed_data)[0]
        
        # Return corresponding mapped category label
        return self.config.CATEGORIES[cluster_idx]

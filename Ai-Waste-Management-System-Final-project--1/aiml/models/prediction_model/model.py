import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class WastePredictor:
    def __init__(self):
        # We use a RandomForest Regressor for time-series forecasting (using lagged features)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, data: pd.DataFrame, target_col: str, feature_cols: list):
        """
        Trains the prediction model.
        data: DataFrame containing historical data.
        """
        X = data[feature_cols]
        y = data[target_col]
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(features)
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load(self, filepath: str):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
        else:
            raise FileNotFoundError(f"Model file {filepath} not found.")

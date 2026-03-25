import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from aiml.utils.logger import get_logger
from aiml.models.prediction_model.model import WastePredictor

logger = get_logger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        pred_config = config.get("model", {}).get("prediction", {})
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    pred_config = {"model_path": "models/prediction_model/rf_model.pkl", "history_days": 30}

class PredictionService:
    def __init__(self):
        logger.info("Initializing PredictionService")
        self.predictor = WastePredictor()
        
        # In a real scenario, load the trained model if available
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / pred_config.get("model_path", "models/prediction_model/rf_model.pkl")
        
        try:
            self.predictor.load(str(model_path))
            logger.info("Loaded pre-trained prediction model.")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Initializing a mock predictor for demonstration.")
            # For demonstration, we simply train it on dummy data so API doesn't crash
            df = pd.DataFrame({
                "day_of_week": [0, 1, 2, 3, 4, 5, 6] * 5,
                "month": [1] * 35,
                "historical_freq": np.random.randint(10, 50, 35),
                "target": np.random.randint(20, 60, 35)
            })
            self.predictor.train(df, "target", ["day_of_week", "month", "historical_freq"])

    def predict_trend(self, historical_data: list) -> dict:
        """
        Takes list of historical data dicts: [{'date': 'YYYY-MM-DD', 'location_id': str, 'complaint_freq': int}]
        Returns future prediction and trend analysis.
        """
        try:
            logger.info(f"Predicting trend for {len(historical_data)} data points")
            
            # Feature engineering for the model
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            
            # Using recent complaint frequency as a simple feature
            # In a real system, you'd aggregate this properly per location.
            recent_freq = df['complaint_freq'].mean()
            
            # Prepare a feature vector for the "next" day
            next_day = df['date'].max() + pd.Timedelta(days=1)
            future_features = pd.DataFrame({
                "day_of_week": [next_day.dayofweek],
                "month": [next_day.month],
                "historical_freq": [recent_freq]
            })
            
            prediction = self.predictor.predict(future_features)[0]
            
            # Simple trend analysis
            avg_past = df['complaint_freq'].mean()
            trend = "increasing" if prediction > avg_past else "decreasing"
            if abs(prediction - avg_past) < (0.05 * avg_past):
                trend = "stable"
                
            result = {
                "predicted_waste_generation": round(prediction, 2),
                "trend": trend,
                "target_date": next_day.strftime("%Y-%m-%d")
            }
            logger.info(f"Prediction complete. Trend: {trend}")
            return result
            
        except Exception as e:
            logger.error(f"Error during trend prediction: {e}")
            raise e

prediction_service = PredictionService()

from fastapi import UploadFile, HTTPException
from aiml.services.image_service import image_service
from aiml.services.geo_service import geo_service
from aiml.services.prediction_service import prediction_service
from aiml.api.schemas import HotspotRequest, TrendRequest
from aiml.utils.logger import get_logger

logger = get_logger(__name__)

async def predict_waste_controller(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        result = image_service.classify_image(content)
        return result
    except Exception as e:
        logger.error(f"Error in predict_waste_controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def detect_hotspot_controller(request: HotspotRequest):
    try:
        coords = [{"lat": c.lat, "lon": c.lon} for c in request.coordinates]
        result = geo_service.process_hotspots(coords)
        return result
    except Exception as e:
        logger.error(f"Error in detect_hotspot_controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def predict_trend_controller(request: TrendRequest):
    try:
        data = [{"date": d.date, "location_id": d.location_id, "complaint_freq": d.complaint_freq} for d in request.historical_data]
        result = prediction_service.predict_trend(data)
        return result
    except Exception as e:
        logger.error(f"Error in predict_trend_controller: {e}")
        raise HTTPException(status_code=500, detail=str(e))

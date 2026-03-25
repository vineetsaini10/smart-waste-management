from fastapi import APIRouter, UploadFile, File
from aiml.api.controllers import predict_waste_controller, detect_hotspot_controller, predict_trend_controller
from aiml.api.schemas import HotspotRequest, TrendRequest

router = APIRouter()

@router.post("/predict-waste", summary="Classify Waste Image")
async def predict_waste(file: UploadFile = File(...)):
    """Upload an image to classify the waste type (wet, dry, plastic, hazardous)."""
    return await predict_waste_controller(file)

@router.post("/detect-hotspot", summary="Detect Complaint Hotspots")
def detect_hotspot(request: HotspotRequest):
    """Pass a list of latitude/longitude coordinates to cluster them into hotspots."""
    return detect_hotspot_controller(request)

@router.post("/predict-trend", summary="Predict Future Waste Trends")
def predict_trend(request: TrendRequest):
    """Provide historical complaint data to predict future trends and waste generation."""
    return predict_trend_controller(request)

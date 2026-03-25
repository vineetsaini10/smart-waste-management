from pydantic import BaseModel, Field
from typing import List

class Coordinate(BaseModel):
    lat: float
    lon: float

class HotspotRequest(BaseModel):
    coordinates: List[Coordinate]

class HistoricalData(BaseModel):
    date: str  # YYYY-MM-DD format
    location_id: str
    complaint_freq: int

class TrendRequest(BaseModel):
    historical_data: List[HistoricalData]

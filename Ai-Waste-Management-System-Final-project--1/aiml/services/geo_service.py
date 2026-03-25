from pathlib import Path
import yaml
from aiml.utils.logger import get_logger
from aiml.models.hotspot_model.model import HotspotDetector

logger = get_logger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        geo_config = config.get("model", {}).get("geo", {})
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    geo_config = {"eps": 0.01, "min_samples": 5}

class GeoSpatialService:
    def __init__(self):
        eps = geo_config.get("eps", 0.01)
        min_samples = geo_config.get("min_samples", 5)
        logger.info(f"Initializing GeoSpatialService with eps={eps}, min_samples={min_samples}")
        self.detector = HotspotDetector(eps=eps, min_samples=min_samples)

    def process_hotspots(self, coordinates: list) -> dict:
        """
        Processes a list of dictionaries [{'lat': float, 'lon': float}]
        and returns clustered hotspots.
        """
        try:
            logger.info(f"Processing {len(coordinates)} coordinate points")
            # Convert to list of lists [lat, lon]
            points = [[pt['lat'], pt['lon']] for pt in coordinates]
            
            result = self.detector.detect_hotspots(points)
            
            # Format output for API
            formatted_clusters = []
            for cluster_id, data in result.get("clusters", {}).items():
                formatted_clusters.append({
                    "cluster_id": cluster_id,
                    "is_noise": data["is_noise"],
                    "points": [{"lat": p[0], "lon": p[1]} for p in data["points"]],
                    "centroid": {"lat": data["centroid"][0], "lon": data["centroid"][1]} if "centroid" in data else None,
                    "intensity": data.get("intensity", 0)
                })
                
            return {"hotspots": formatted_clusters}
            
        except Exception as e:
            logger.error(f"Error during hotspot detection: {e}")
            raise e

geo_service = GeoSpatialService()

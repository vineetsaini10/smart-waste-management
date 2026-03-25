from sklearn.cluster import DBSCAN
import numpy as np

class HotspotDetector:
    def __init__(self, eps: float = 0.01, min_samples: int = 5):
        """
        Initializes DBSCAN for clustering geospatial coordinates (lat, lon).
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
             For lat/lon, 0.01 roughly equals 1km.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        
    def detect_hotspots(self, coordinates: list) -> dict:
        """
        Takes a list of [lat, lon] coordinates and returns cluster assignments.
        """
        if not coordinates:
            return {"clusters": {}}
            
        X = np.array(coordinates)
        labels = self.model.fit_predict(X)
        
        # Group points by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            # label -1 means noise
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = {"points": [], "is_noise": cluster_id == -1}
            clusters[cluster_id]["points"].append(coordinates[idx])
            
        # Optional: calculate centroid for each valid cluster
        for cluster_id, data in clusters.items():
            if not data["is_noise"] and data["points"]:
                pts = np.array(data["points"])
                centroid = pts.mean(axis=0)
                data["centroid"] = centroid.tolist()
                data["intensity"] = len(data["points"])
                
        return {"clusters": clusters}

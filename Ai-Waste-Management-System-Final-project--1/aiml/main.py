from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

# Add project root to python path to allow absolute imports like `aiml.api`
sys.path.append(str(Path(__file__).parent.parent))

from aiml.api.routes import router
from aiml.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Waste Management Backend",
    description="ML backend for Waste Image Classification, Hotspot Detection, and Trend Prediction",
    version="1.0.0"
)

# CORS setup for frontend/Node.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "service": "aiml-backend"}

if __name__ == "__main__":
    logger.info("Starting AI/ML backend server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

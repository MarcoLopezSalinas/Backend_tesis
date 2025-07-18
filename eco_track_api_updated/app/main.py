from fastapi import FastAPI, HTTPException
from .schemas import RecommendationRequest, ItineraryResponse
from .services import get_itineraries

app = FastAPI(title="EcoTrack Recommendations API")

@app.post("/itineraries", response_model=ItineraryResponse)
async def itineraries_endpoint(req: RecommendationRequest):
    try:
        return get_itineraries(req)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

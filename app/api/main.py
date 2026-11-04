# FastAPI app for the air target identifier

import os
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.features import ENGINE_TYPES, IR_CATEGORIES
from .service import MAX_TOP_K, Service

service = Service.load()
app = FastAPI(title="Air Target Identification API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip().rstrip("/") for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Observation(BaseModel):
    speed: float | None = Field(None, ge=0, le=40000, description="km/h")
    altitude: float | None = Field(None, ge=0, le=200000, description="m")
    climb_rate: float | None = Field(None, ge=0, le=5000, description="m/s, magnitude")
    turn_rate: float | None = Field(None, ge=0, le=200, description="deg/s")
    rcs: float | None = Field(None, gt=0, le=10000, description="m²")
    engine_type: Literal[tuple(ENGINE_TYPES)] | None = None
    engine_count: int | None = Field(None, ge=0, le=10)
    afterburner: Literal["yes", "no"] | None = None
    ir_category: Literal[tuple(IR_CATEGORIES)] | None = None
    length_m: float | None = Field(None, gt=0, le=100)
    wingspan_m: float | None = Field(None, gt=0, le=100)
    mtow_kg: float | None = Field(None, gt=0, le=700000)
    first_flight: int | None = Field(None, ge=1900, le=2100)


class PredictRequest(BaseModel):
    scenario: str
    observations: list[Observation] = Field(..., min_length=1, max_length=50)
    countries: list[str] | None = None
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)


@app.get("/api/health")
def health():
    body = {"mode": service.mode, "models": len(service.models)}
    if not service.models:
        return JSONResponse({"status": "no models loaded", **body}, status_code=503)
    return {"status": "ok", **body}


@app.get("/api/meta")
def meta():
    return {
        "mode": service.mode,
        "scenarios": service.scenarios(),
        "engine_types": ENGINE_TYPES,
        "ir_categories": list(IR_CATEGORIES),
        "countries": service.countries(),
        "max_top_k": MAX_TOP_K,
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    observations = [o.model_dump(exclude_none=True) for o in request.observations]
    if not any(observations):
        raise HTTPException(422, "Every snapshot is empty: enter at least one measured value")
    try:
        return service.predict(request.scenario, observations, request.countries, request.top_k)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

# app/lead_score/models.py
from typing import TypedDict, List
from pydantic import BaseModel, Field

class LeadScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Lead score as float between 0 and 1")
    explanation: str
    filename: str

class LeadAnalysisResult(TypedDict):
    transcription: str
    score: float
    explanation: str

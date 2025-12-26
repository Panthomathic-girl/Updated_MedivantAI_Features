# app/lead_score/views.py
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.lead_score.models import LeadScoreResponse
from app.lead_score.agent import lead_score_agent

router = APIRouter(prefix="/lead_score", tags=["Lead Scoring"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Audio extensions & MIME types
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
AUDIO_MIME = {
    "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/flac", "audio/aac", "audio/ogg", "audio/webm", "audio/m4a"
}

# Video extensions & MIME types
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
VIDEO_MIME = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
    "video/webm", "video/mpeg", "video/x-m4v"
}

ALLOWED_EXT = AUDIO_EXT.union(VIDEO_EXT)
ALLOWED_MIME = AUDIO_MIME.union(VIDEO_MIME)

MAX_SIZE_MB = 150
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

@router.post("/", response_model=LeadScoreResponse)
async def predict_lead_score(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Invalid MIME: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_SIZE_MB}MB)")

    file_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        temp_path.write_bytes(content)
        result = await lead_score_agent(str(temp_path), file.content_type)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to parse LLM response")

        score_float = float(result.get("score", 0))

        return LeadScoreResponse(
            score=score_float,
            explanation=result.get("explanation", "No explanation provided."),
            filename=filename
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
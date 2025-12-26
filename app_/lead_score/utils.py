# app/lead_score/utils.py
from typing import Any, Optional
from app.lead_score.models import LeadAnalysisResult
import json
import re

# Optional: If you're using actual protobuf objects from google-generativeai
try:
    from google.generativeai.types import GenerateContentResponse
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False



def parse_gemini_generate_content_response(response: Any) -> Optional[LeadAnalysisResult]:
    text_content = ""

    if PROTO_AVAILABLE and isinstance(response, GenerateContentResponse):
        if response.candidates and response.candidates[0].content.parts:
            text_content = response.candidates[0].content.parts[0].text

    elif isinstance(response, dict):
        candidates = response.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                text_content = parts[0]["text"]

    elif isinstance(response, str):
        json_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return {
                    "transcription": parsed.get("transcription", ""),
                    "score": parsed.get("score", 0),
                    "explanation": parsed.get("explanation", "")
                }
            except json.JSONDecodeError:
                pass

        text_match = re.search(r'"text":\s*"([^"]+)"', response.replace('\n', ' '))
        if text_match:
            text_content = text_match.group(1).replace("\\n", "\n")

    if text_content.strip():
        json_block_match = re.search(r"```json\s*({.*?})\s*```", text_content, re.DOTALL)
        if json_block_match:
            try:
                data = json.loads(json_block_match.group(1))
                return {
                    "transcription": data.get("transcription", ""),
                    "score": float(data.get("score", 0)),
                    "explanation": data.get("explanation", "")
                }
            except (json.JSONDecodeError, ValueError):
                return None

        try:
            data = json.loads(text_content.strip())
            return {
                "transcription": data.get("transcription", ""),
                "score": float(data.get("score", 0)),
                "explanation": data.get("explanation", "")
            }
        except json.JSONDecodeError:
            pass

    return None
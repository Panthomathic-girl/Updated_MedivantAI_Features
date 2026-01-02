# app/predictive_lead_score/services.py
import json
from pathlib import Path
from google import generativeai as genai
from google.generativeai import protos
from typing import Any, Dict, Optional
from config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

def analyze_audio_file(filepath: str, mime_type: str, prompt: str) -> dict:
    model = genai.GenerativeModel(GEMINI_MODEL)

    part = protos.Part(
        inline_data=protos.Blob(
            mime_type=mime_type,
            data=Path(filepath).read_bytes()
        )
    )

    response = model.generate_content([prompt, part])
    return response
        
def generate_text_response(prompt: str) -> str:

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_structured_json(
    prompt: str,
    response_schema: Any,
    temperature: float = 0.2
) -> Dict[str, Any]:

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            # safety_settings=RELAXED_SAFETY
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
            ),
        )

        # Native JSON mode success
        if response.text:
            return json.loads(response.text)
        
        return response
            
    except Exception as e:
        print(f"JSON mode failed ({e}), falling back to text mode...")
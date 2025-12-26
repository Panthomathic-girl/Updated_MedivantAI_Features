# app/smart_task/utils.py
from email.mime import text
import json
import re
import logging
from typing import Optional, Dict, Any, get_args
from app.smart_task.models import SmartTaskResponse, SuggestedTaskForm
from config import SmartTaskConfig

logger = logging.getLogger(__name__)

def extract_json_from_llm_response(text: str) -> Optional[Dict]:
    if not text:
        return None

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    try:
        cleaned = text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)
    except:
        pass

    rough = re.search(r'\{.*"suggested_tasks"\s*:\s*\[.*\].*\}', text, re.DOTALL)
    if rough:
        try:
            return json.loads(rough.group(0))
        except:
            pass

    return None

def _normalize_task_type(task_type: str) -> str:
    task_type_lower = task_type.strip().lower()
    return SmartTaskConfig.TASK_TYPE_MAPPING.get(task_type_lower, task_type_lower)

# New: Strict date pattern
DATE_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")

def _sanitize_date(value: Any) -> Optional[str]:
    """Return valid dd/mm/yyyy string or None"""
    if value is None or value == "null":
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        # Reject common non-specific values
        vague_words = {"today", "tomorrow", "asap", "soon", "next week", "next month", "tbd", "pending"}
        if value in vague_words or value == "":
            return None
        # Only allow exact dd/mm/yyyy
        if DATE_PATTERN.match(value):
            return value  # Keep as-is
    return None

def parse_structured_tasks(response: Any) -> SmartTaskResponse:
    try:
        text = ""
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            part = response.candidates[0].content.parts[0]
            text = getattr(part, "text", "")
        elif isinstance(response, str):
            text = response

        if not text:
            return SmartTaskResponse(suggested_tasks=[], call_has_next_steps=False, summary="No content detected.")

        data = extract_json_from_llm_response(text)
        if not data:
            return SmartTaskResponse(suggested_tasks=[], call_has_next_steps=False, summary="Invalid JSON from AI.")

        raw_tasks = data.get("suggested_tasks", [])
        summary = data.get("summary", "Next steps generated.")

        validated_tasks = []
        VALID_TASK_TYPES = set(get_args(SmartTaskConfig.TASKTYPE))

        for task_data in raw_tasks[:5]:
            if not isinstance(task_data, dict):
                continue

            # Normalize task_type
            if "task_type" in task_data:
                original = task_data["task_type"]
                normalized = _normalize_task_type(original)
                if normalized in VALID_TASK_TYPES:
                    task_data["task_type"] = normalized
                else:
                    task_data["task_type"] = "follow up" if "follow" in normalized else "administrative"

            # Sanitize dates: only keep valid dd/mm/yyyy, else null
            if "due_date" in task_data:
                task_data["due_date"] = _sanitize_date(task_data["due_date"])
            if "reminder_date" in task_data:
                task_data["reminder_date"] = _sanitize_date(task_data["reminder_date"])

            # Optional: clean up other optional fields
            for field in ["related_to", "created_by", "notes"]:
                if field in task_data and task_data[field] in [None, ""]:
                    task_data[field] = None if field != "notes" else ""

            try:
                task = SuggestedTaskForm(**task_data)
                validated_tasks.append(task)
            except Exception as e:
                logger.warning(f"Task validation failed after sanitization: {e} | Original: {task_data}")
                continue

        has_tasks = len(validated_tasks) > 0

        return SmartTaskResponse(
            suggested_tasks=validated_tasks,
            call_has_next_steps=has_tasks,
            summary=summary if has_tasks else "No actionable next steps identified."
        )

    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return SmartTaskResponse(suggested_tasks=[], call_has_next_steps=False, summary="Processing failed.")
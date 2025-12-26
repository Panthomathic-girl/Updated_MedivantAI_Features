# app/smart_task/agent.py
from app.llm import analyze_audio_file
from app.smart_task.utils import parse_structured_tasks
from config import SmartTaskConfig
from typing import get_args

def suggest_next_tasks(file_path: str, mime_type: str) -> dict:
    allowed_task_types = ", ".join(f'"{t}"' for t in get_args(SmartTaskConfig.TASKTYPE))

    prompt = f"""
You are the world's best B2B sales assistant analyzing a sales call (audio or video).

CRITICAL RULES:

1. TASK TYPES MUST BE EXACTLY ONE OF THESE ONLY:
{allowed_task_types}

2. For due_date and reminder_date:
   - Use format dd/mm/yyyy (e.g., 25/12/2025) if a specific date is mentioned.
   - If vague (e.g., "today", "tomorrow", "next week", "ASAP", "soon"), leave as null (do not guess).
   - If no date mentioned at all, use null.

Return ONLY valid JSON. No markdown. No explanations.

Example with specific dates:
{{
  "suggested_tasks": [
    {{
      "task_title": "Send updated proposal",
      "task_type": "email",
      "related_to": "Acme Corp - Q4 Deal",
      "due_date": "10/01/2026",
      "priority": "high",
      "status": "not started",
      "reminder_date": "08/01/2026",
      "notes": "Client requested revised pricing by early January."
    }}
  ],
  "summary": "Positive call. Client wants proposal update."
}}

Example with no/vague dates:
{{
  "suggested_tasks": [
    {{
      "task_title": "Follow up with decision maker",
      "task_type": "call",
      "related_to": "Acme Corp - Q4 Deal",
      "due_date": null,
      "priority": "high",
      "status": "not started",
      "reminder_date": null,
      "notes": "Client said 'call me next week' — no specific date given."
    }}
  ],
  "summary": "Interest shown. Follow up needed."
}}

If no next steps:
{{
  "suggested_tasks": [],
  "call_has_next_steps": false,
  "summary": "No actionable outcomes or interest."
}}

Analyze the call and return JSON only.
"""

    raw_response = analyze_audio_file(file_path, mime_type, prompt)
    parsed_result = parse_structured_tasks(raw_response)
    return parsed_result.dict()
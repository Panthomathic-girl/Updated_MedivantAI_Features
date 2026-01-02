# config.py
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    PORT_NO: int = int(os.getenv("PORT_NO", "8000"))
    

    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required in .env")

class ModelConfig:
    LEAD_MODEL_FILE = "app/predictive_lead_score/models/deal_closure_model.pkl"
    ORDER_FORECAST_MODEL_FILE = Path("app/order_forecasting/models/order_forecast_model.pkl") 
    ENRICHMENT_FAISS_INDEX = Path("app/bulk_upload/models/faiss_leads_index.faiss")
    ENRICHMENT_META_FILE = Path("app/bulk_upload/models/faiss_leads_meta.pkl")
    ENRICHMENT_ID_TO_IDX_PATH = Path("app/bulk_upload/models/faiss_id_to_idx.pkl")
    ENRICHMENT_SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
class SmartTaskConfig:
    TASKTYPE = Literal["call", "email", "meeting", "administrative", "follow up"]
    PriorityType = Literal["low", "medium", "high", "urgent"]
    StatusType = Literal["not started", "in progress", "completed", "waiting", "deferred"]
    
settings = Settings()
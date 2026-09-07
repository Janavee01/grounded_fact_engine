from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class FactType(str, Enum):
    NUMERIC = "numeric"
    SEMANTIC = "semantic"
    DATE = "date"
    BOOLEAN = "boolean"
    ENTITY = "entity"

class Fact(BaseModel):
    id: str
    text: str
    fact_type: FactType
    value: Optional[Any] = None
    unit: Optional[str] = None
    context: Dict[str, Any] = {}
    source_document: str
    source_page: int
    source_snippet: str
    confidence: float = 0.0
    extraction_method: str = "llm"
    created_at: datetime = datetime.now()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FactComparison(BaseModel):
    fact1_id: str
    fact2_id: str
    relationship: str  # "corroborates", "contradicts", "unrelated"
    confidence: float
    explanation: str
    context_notes: Optional[str] = None
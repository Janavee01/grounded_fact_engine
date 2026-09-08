# src/models/fact.py

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


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

    context: Dict[str, Any] = Field(
        default_factory=dict
    )

    source_document: str
    source_page: int
    source_snippet: str

    confidence: float = 0.0
    extraction_method: str = "llm"

    created_at: datetime = Field(
        default_factory=datetime.now
    )

    class Config:
        json_encoders = {
            datetime: lambda value: value.isoformat()
        }


class FactComparison(BaseModel):
    fact1_id: str
    fact2_id: str

    relationship: str

    confidence: float
    explanation: str

    context_notes: Optional[str] = None
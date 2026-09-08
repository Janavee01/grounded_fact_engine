from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

Base = declarative_base()

class FactRecord(Base):
    __tablename__ = 'facts'
    
    id = Column(String, primary_key=True)
    text = Column(String)
    fact_type = Column(String)
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    context = Column(JSON)
    source_document = Column(String)
    source_page = Column(Integer)
    source_snippet = Column(String)
    confidence = Column(Float)
    extraction_method = Column(String)
    created_at = Column(DateTime, default=datetime.now)

class Database:
    def __init__(self, db_path="data/facts.db"):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_fact(self, fact):
        """Save a fact to database"""
        session = self.Session()
        try:
            record = FactRecord(
                id=fact.id,
                text=fact.text,
                fact_type=fact.fact_type.value if hasattr(fact.fact_type, 'value') else str(fact.fact_type),
                value=fact.value,
                unit=fact.unit,
                context=fact.context,
                source_document=fact.source_document,
                source_page=fact.source_page,
                source_snippet=fact.source_snippet,
                confidence=fact.confidence,
                extraction_method=fact.extraction_method
            )
            session.add(record)
            session.commit()
            return fact.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_all_facts(self):
        """Get all facts"""
        session = self.Session()
        try:
            records = session.query(FactRecord).all()
            return [self._record_to_dict(r) for r in records]
        finally:
            session.close()
    
    def get_facts_by_document(self, document_name):
        """Get facts from a specific document"""
        session = self.Session()
        try:
            records = session.query(FactRecord).filter_by(source_document=document_name).all()
            return [self._record_to_dict(r) for r in records]
        finally:
            session.close()
    
    def clear_all_facts(self):
        """Delete all facts from the database"""
        session = self.Session()
        try:
            count = session.query(FactRecord).delete()
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _record_to_dict(self, record):
        """Convert SQLAlchemy record to dict"""
        return {
            'id': record.id,
            'text': record.text,
            'fact_type': record.fact_type,
            'value': record.value,
            'unit': record.unit,
            'context': record.context,
            'source_document': record.source_document,
            'source_page': record.source_page,
            'source_snippet': record.source_snippet,
            'confidence': record.confidence,
            'created_at': record.created_at.isoformat() if record.created_at else None
        }
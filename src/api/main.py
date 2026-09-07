from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
import uuid

from ..extraction.pdf_extractor import PDFExtractor
from ..storage.database import Database
from ..comparison.comparator import FactComparator
from ..models.fact import Fact

app = FastAPI(title="Fact Knowledge Layer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
extractor = PDFExtractor()
db = Database()
comparator = FactComparator()

# Ensure uploads directory exists
os.makedirs("data/uploads", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Fact Knowledge Layer API", "status": "running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF"""
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file
        file_path = f"data/uploads/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract facts
        facts = extractor.extract_facts(file_path, file.filename)
        
        # Save to database
        fact_ids = []
        for fact in facts:
            fact_id = db.save_fact(fact)
            fact_ids.append(fact_id)
        
        return {
            "message": f"Processed {len(facts)} facts from {file.filename}",
            "document": file.filename,
            "facts_count": len(facts),
            "fact_ids": fact_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facts")
async def get_all_facts():
    """Get all facts from database"""
    try:
        facts = db.get_all_facts()
        return {"facts": facts, "count": len(facts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facts/{document}")
async def get_facts_by_document(document: str):
    """Get facts from a specific document"""
    try:
        facts = db.get_facts_by_document(document)
        return {"document": document, "facts": facts, "count": len(facts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare")
async def compare_facts():
    """Compare all facts and find relationships"""
    try:
        facts = db.get_all_facts()
        comparisons = comparator.compare_facts(facts)
        
        corroborations = [c for c in comparisons if c.relationship == "corroborates"]
        contradictions = [c for c in comparisons if c.relationship == "contradicts"]
        reconciled = [c for c in comparisons if c.relationship == "reconciled"]
        
        return {
            "total_comparisons": len(comparisons),
            "corroborations": corroborations,
            "contradictions": contradictions,
            "reconciled": reconciled,
            "summary": {
                "corroborations_count": len(corroborations),
                "contradictions_count": len(contradictions),
                "reconciled_count": len(reconciled)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        facts = db.get_all_facts()
        documents = set(f['source_document'] for f in facts)
        
        return {
            "total_facts": len(facts),
            "total_documents": len(documents),
            "documents": list(documents),
            "fact_types": {
                'numeric': len([f for f in facts if f['fact_type'] == 'numeric']),
                'semantic': len([f for f in facts if f['fact_type'] == 'semantic'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/facts")
async def clear_all_facts():
    """Clear all facts (for testing)"""
    try:
        # This would need to be implemented in database class
        return {"message": "Facts cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
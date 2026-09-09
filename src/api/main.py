from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
import uuid
from threading import Lock

from starlette.concurrency import run_in_threadpool

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

compare_progress = {
    "running": False,
    "completed": 0,
    "total": 0,
    "stage": "Waiting to start",
    "error": None,
}
compare_progress_lock = Lock()

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
        total_pairs = len(facts) * (len(facts) - 1) // 2

        def update_progress(completed: int, total: int, stage: str):
            with compare_progress_lock:
                compare_progress.update(
                    running=True,
                    completed=completed,
                    total=total,
                    stage=stage,
                    error=None,
                )

        update_progress(0, total_pairs, "Starting comparison")
        comparisons = await run_in_threadpool(
            comparator.compare_facts,
            facts,
            update_progress,
        )
        
        corroborations = [c for c in comparisons if c.relationship == "corroborates"]
        contradictions = [c for c in comparisons if c.relationship == "contradicts"]
        reconciled = [c for c in comparisons if c.relationship == "reconciled"]
        
        result = {
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
        with compare_progress_lock:
            compare_progress.update(
                running=False,
                completed=total_pairs,
                total=total_pairs,
                stage="Comparison complete",
            )
        return result
    except Exception as e:
        with compare_progress_lock:
            compare_progress.update(running=False, stage="Comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare/progress")
async def get_compare_progress():
    """Return progress for the active comparison run."""
    with compare_progress_lock:
        return dict(compare_progress)

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
        db = Database()
        count = db.clear_all_facts()
        return {"message": "Facts cleared", "deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

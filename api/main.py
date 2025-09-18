import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Import the new FileProcessor class
from src.document_ingestion.data_ingestion import (
    FileProcessor, # Use FileProcessor instead of DocHandler and DocumentComparator
    ChatIngestor,
    FaissManager, # FaissManager is still relevant
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter # read_pdf_via_handler is no longer needed

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        # Use FileProcessor instead of DocHandler
        fp = FileProcessor(base_dir=os.path.join(UPLOAD_BASE, "document_analysis"))
        
        # Save the uploaded file (adapting FastAPI's UploadFile)
        saved_path = fp.save_file(FastAPIFileAdapter(file))
        
        # Read the content using the generic read_file method
        text = fp.read_file(saved_path)
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        # Use FileProcessor for document comparison
        fp = FileProcessor(base_dir=os.path.join(UPLOAD_BASE, "document_compare"))
        
        # Save both reference and actual files
        ref_path = fp.save_file(FastAPIFileAdapter(reference))
        act_path = fp.save_file(FastAPIFileAdapter(actual))
        
        # Combine the text content of all files in the session (which are ref and act)
        combined_text = fp.combine_documents_in_session()
        
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from one or both documents for comparison.")

        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient="records"), "session_id": fp.session_id}
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # Assuming ChatIngestor.built_retriver now uses FileProcessor internally as updated previously
        ci.built_retriver(
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Indexing failed: {e}") # Log the error
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
        response = rag.invoke(question, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Query failed: {e}") # Log the error
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


if __name__ == "__main__":
    import uvicorn
    # Make sure to run from the project root or adjust the import path for uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True) # Changed to 0.0.0.0 and port 8080 for broader access
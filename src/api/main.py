from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import json

from src.rag_pipeline import RAGPipeline
from src.config import settings

app = FastAPI(
    title="RAG Document Assistant",
    version="1.0.0",
)

pipeline = RAGPipeline(
    collection_name=settings.chroma_collection,
    persist_directory=settings.chroma_persist_dir,
    model=settings.ollama_model,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    top_k=settings.top_k,
)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    filename: str
    doc_id: str
    chunks: int
    type: str


class StatsResponse(BaseModel):
    vector_store: dict
    ollama_connected: bool
    available_models: list[str]
    current_model: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    return pipeline.get_stats()


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix not in pipeline.loader.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {pipeline.loader.SUPPORTED_EXTENSIONS}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest_file(tmp_path)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/query")
async def query(request: QueryRequest):
    if not pipeline.llm.check_connection():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with 'ollama serve'"
        )

    if request.stream:
        return StreamingResponse(
            stream_response(request.question, request.top_k),
            media_type="text/event-stream"
        )

    try:
        result = pipeline.query(request.question, top_k=request.top_k, stream=False)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(question: str, top_k: Optional[int]):
    try:
        for chunk in pipeline.query(question, top_k=top_k, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.delete("/documents")
async def clear_documents():
    pipeline.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

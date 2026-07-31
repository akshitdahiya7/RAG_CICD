from fastapi import FastAPI
from pydantic import BaseModel

from ingest.catalog_parser import CatalogParser
from ingest.embedder import Embedder
from ingest.fetch_catalog import DEST as CATALOG_PATH, fetch as fetch_catalog
from ingest.vector_store import QdrantStore

app = FastAPI(title="Compliance Control Assistant")

# Loaded once at import time (app startup), reused across every request —
# loading the embedding model is expensive, so it must not happen per-request.
embedder = Embedder()
vector_store = QdrantStore(embedder=embedder)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchHit(BaseModel):
    control_id: str
    title: str
    score: float
    text: str


class IngestResponse(BaseModel):
    controls_indexed: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    if not CATALOG_PATH.exists():
        fetch_catalog()

    controls = CatalogParser().parse(CATALOG_PATH)
    vector_store.upsert_controls(controls)
    return IngestResponse(controls_indexed=len(controls))


@app.post("/query", response_model=list[SearchHit])
def query(request: QueryRequest):
    results = vector_store.search(request.question, top_k=request.top_k)
    return [
        SearchHit(
            control_id=control.id,
            title=control.title,
            score=score,
            text=control.to_chunk_text(),
        )
        for control, score in results
    ]

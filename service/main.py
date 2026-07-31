from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()  # must run before any module below reads OPENAI_API_KEY at import time

from ingest.catalog_parser import CatalogParser
from ingest.embedder import Embedder
from ingest.fetch_catalog import DEST as CATALOG_PATH, fetch as fetch_catalog
from ingest.vector_store import QdrantStore
from service.generator import Generator

app = FastAPI(title="Compliance Control Assistant")

# Loaded once at import time (app startup), reused across every request —
# loading the embedding model is expensive, so it must not happen per-request.
embedder = Embedder()
vector_store = QdrantStore(embedder=embedder)
generator = Generator()


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


class QueryResponse(BaseModel):
    answer: str
    sources: list[SearchHit]


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


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    results = vector_store.search(request.question, top_k=request.top_k)
    sources = [
        SearchHit(
            control_id=control.id,
            title=control.title,
            score=score,
            text=control.to_chunk_text(),
        )
        for control, score in results
    ]
    answer = generator.generate(request.question, [control for control, _ in results])
    return QueryResponse(answer=answer, sources=sources)

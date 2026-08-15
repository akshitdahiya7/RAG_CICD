import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()  # must run before any module below reads OPENAI_API_KEY/LANGFUSE_* at import time

from langfuse import get_client, observe

from ingest.catalog_parser import CatalogParser
from ingest.embedder import Embedder
from ingest.fetch_catalog import DEST as CATALOG_PATH, fetch as fetch_catalog
from ingest.vector_store import QdrantStore
from service.generator import Generator

app = FastAPI(title="Compliance Control Assistant")

# Exposes GET /metrics with request count, latency histograms, and in-progress
# request gauges out of the box — Prometheus scrapes that endpoint (see
# infra/prometheus.yml), Grafana queries Prometheus.
Instrumentator().instrument(app).expose(app)

# Loaded once at import time (app startup), reused across every request —
# loading the embedding model is expensive, so it must not happen per-request.
embedder = Embedder()
vector_store = QdrantStore(
    embedder=embedder,
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)
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
@observe(name="rag_query")
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

    # The generator's own OpenAI call is already traced automatically (see
    # generator.py). This adds the retrieval step to the same trace, since
    # that's the other half of "why did the RAG system answer this way" —
    # which controls got retrieved, and how relevant they were.
    get_client().update_current_span(
        metadata={
            "top_k": request.top_k,
            "retrieved_control_ids": [s.control_id for s in sources],
            "retrieval_scores": [s.score for s in sources],
        }
    )

    answer = generator.generate(request.question, [control for control, _ in results])
    return QueryResponse(answer=answer, sources=sources)

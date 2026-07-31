import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingest.embedder import Embedder
from ingest.models import Control

# Deterministic namespace so uuid5(NAMESPACE, "ac-1") is the same point id
# every run — re-ingesting updates the existing point instead of duplicating it.
POINT_ID_NAMESPACE = uuid.UUID("d6e30f6e-6e8f-4b0e-9c1a-3f6b0f2a2b11")


def control_point_id(control_id: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, control_id))


class QdrantStore:
    def __init__(self, embedder: Embedder, host: str = "localhost", port: int = 6333, collection_name: str = "nist_800_53_controls"):
        self.client = QdrantClient(host=host, port=port)
        self.embedder = embedder
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedder.dimension, distance=Distance.COSINE),
            )

    def upsert_controls(self, controls: list[Control]) -> None:
        self.ensure_collection()
        texts = [control.to_chunk_text() for control in controls]
        vectors = self.embedder.embed(texts)

        points = [
            PointStruct(
                id=control_point_id(control.id),
                vector=vector,
                payload={
                    "control_id": control.id,
                    "family_id": control.family_id,
                    "family_title": control.family_title,
                    "title": control.title,
                    "statement": control.statement,
                    "guidance": control.guidance,
                    "assessment_objectives": control.assessment_objectives,
                    "is_withdrawn": control.is_withdrawn,
                },
            )
            for control, vector in zip(controls, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Control, float]]:
        query_vector = self.embedder.embed([query])[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        results = []
        for hit in search_result:
            payload = hit.payload
            control = Control(
                id=payload["control_id"],
                family_id=payload["family_id"],
                family_title=payload["family_title"],
                title=payload["title"],
                statement=payload["statement"],
                guidance=payload["guidance"],
                assessment_objectives=payload["assessment_objectives"],
                is_withdrawn=payload["is_withdrawn"],
            )
            results.append((control, hit.score))
        return results

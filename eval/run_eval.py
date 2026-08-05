import torch  # noqa: F401 — must import before ragas/sentence_transformers pull it in
                          # indirectly, or torch's DLL init fails intermittently on Windows

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import mlflow
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from ingest.embedder import Embedder
from ingest.vector_store import QdrantStore
from service.generator import Generator

GOLDEN_SET_PATH = Path(__file__).parent / "golden_dataset.json"

TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-small-en"
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"


def build_samples(
    golden_rows: list[dict], vector_store: QdrantStore, generator: Generator
) -> list[SingleTurnSample]:
    samples = []
    for row in golden_rows:
        question = row["question"]
        ground_truth = row["ground_truth"]

        # Run every row (answerable AND unanswerable) through the real
        # pipeline — the unanswerable rows exist specifically to test
        # whether real retrieval + generation decline correctly, so they
        # need to actually hit the generator, not a hardcoded stand-in.
        results = vector_store.search(question, top_k=TOP_K)
        controls = [control for control, _ in results]
        retrieved_contexts = [control.to_chunk_text() for control in controls]
        answer = generator.generate(question, controls)

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                response=answer,
                reference=ground_truth,
            )
        )
    return samples


def main() -> None:
    golden_rows = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    embedder = Embedder(EMBEDDING_MODEL)
    vector_store = QdrantStore(embedder=embedder)
    generator = Generator(model=JUDGE_MODEL)

    samples = build_samples(golden_rows, vector_store, generator)
    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_factory(JUDGE_MODEL),
        embeddings=embedding_factory(JUDGE_EMBEDDING_MODEL),
    )

    mlflow.set_experiment("compliance-rag-eval")
    with mlflow.start_run():
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("generator_model", generator.model)
        mlflow.log_param("golden_set_size", len(golden_rows))

        scores = result._repr_dict
        for metric_name, score in scores.items():
            mlflow.log_metric(metric_name, score)

        mlflow.log_table(result.to_pandas(), artifact_file="eval_results.json")

    print(scores)


if __name__ == "__main__":
    main()

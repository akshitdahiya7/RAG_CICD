import torch  # noqa: F401 — must import before ragas/sentence_transformers pull it in
                          # indirectly, or torch's DLL init fails intermittently on Windows

import json
import sys
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

# Minimum acceptable RAGAS scores — only meaningful for answerable questions
# (see check_refusals below for why unanswerable ones are graded separately).
MIN_SCORES = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.6,
    "context_precision": 0.6,
    "context_recall": 0.6,
}

# Simple substring check for "the model declined instead of guessing" — no LLM
# judge needed, and RAGAS's metrics don't score refusal-as-correct-behavior
# well (confirmed empirically: they scored inconsistently across three nearly
# identical refusal responses in earlier manual testing).
REFUSAL_MARKERS = ["do not contain", "does not contain", "not covered", "no information"]


def run_pipeline(question: str, vector_store: QdrantStore, generator: Generator) -> tuple[list, str]:
    """Run one question through the real retrieval + generation pipeline.
    Returns (retrieved Control objects, generated answer)."""
    results = vector_store.search(question, top_k=TOP_K)
    controls = [control for control, _ in results]
    answer = generator.generate(question, controls)
    return controls, answer


def build_samples(
    golden_rows: list[dict], vector_store: QdrantStore, generator: Generator
) -> list[SingleTurnSample]:
    samples = []
    for row in golden_rows:
        controls, answer = run_pipeline(row["question"], vector_store, generator)
        retrieved_contexts = [control.to_chunk_text() for control in controls]
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=retrieved_contexts,
                response=answer,
                reference=row["ground_truth"],
            )
        )
    return samples


def check_refusals(
    golden_rows: list[dict], vector_store: QdrantStore, generator: Generator
) -> tuple[int, int, list[str]]:
    """For unanswerable questions, check whether the real pipeline declined
    rather than hallucinated. Returns (num_correct, num_total, failure descriptions)."""
    correct = 0
    failures = []
    for row in golden_rows:
        _, answer = run_pipeline(row["question"], vector_store, generator)
        declined = any(marker in answer.lower() for marker in REFUSAL_MARKERS)
        if declined:
            correct += 1
        else:
            failures.append(f"'{row['question']}' -> did not decline: {answer[:100]}")
    return correct, len(golden_rows), failures


def main() -> None:
    golden_rows = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    answerable_rows = [row for row in golden_rows if row["answerable"]]
    unanswerable_rows = [row for row in golden_rows if not row["answerable"]]

    embedder = Embedder(EMBEDDING_MODEL)
    vector_store = QdrantStore(embedder=embedder)
    generator = Generator(model=JUDGE_MODEL)

    samples = build_samples(answerable_rows, vector_store, generator)
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_factory(JUDGE_MODEL),
        embeddings=embedding_factory(JUDGE_EMBEDDING_MODEL),
    )
    scores = result._repr_dict

    refusal_correct, refusal_total, refusal_failures = check_refusals(
        unanswerable_rows, vector_store, generator
    )

    mlflow.set_experiment("compliance-rag-eval")
    with mlflow.start_run():
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("generator_model", generator.model)
        mlflow.log_param("answerable_set_size", len(answerable_rows))
        mlflow.log_param("unanswerable_set_size", len(unanswerable_rows))

        for metric_name, score in scores.items():
            mlflow.log_metric(metric_name, score)
        mlflow.log_metric("refusal_rate", refusal_correct / refusal_total if refusal_total else 1.0)

        mlflow.log_table(result.to_pandas(), artifact_file="eval_results.json")

    print("RAGAS scores (answerable questions):", scores)
    print(f"Refusal check (unanswerable questions): {refusal_correct}/{refusal_total} correctly declined")

    gate_failures = [
        f"{metric}: {scores[metric]:.3f} < {min_score}"
        for metric, min_score in MIN_SCORES.items()
        if scores.get(metric, 0) < min_score
    ]
    if refusal_correct < refusal_total:
        gate_failures.append(f"refusal check: {refusal_correct}/{refusal_total} — see below")

    if gate_failures:
        print("\nQUALITY GATE FAILED:")
        for failure in gate_failures:
            print(" -", failure)
        for failure in refusal_failures:
            print("   ", failure)
        sys.exit(1)

    print("\nQuality gate passed.")


if __name__ == "__main__":
    main()

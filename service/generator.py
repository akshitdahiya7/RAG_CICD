import os

from openai import OpenAI

from ingest.models import Control


SYSTEM_PROMPT = (
    "You are a compliance assistant answering questions about NIST SP 800-53 "
    "security controls. Answer ONLY using the provided control excerpts. "
    "If the excerpts don't contain the answer, say so explicitly instead of guessing."
)


class Generator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, question: str, contexts: list[Control]) -> str:
        context_block = "\n\n".join(
            f"[{i}] {control.to_chunk_text()}" for i, control in enumerate(contexts, start=1)
        )
        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content


import os
from google import genai

SYSTEM = """You are a grounded RAG assistant.

Rules:
1. Answer ONLY using the provided CONTEXT.
2. Never invent or assume information.
3. If the answer is not supported by the CONTEXT, say:
"I don't have enough information in the provided documents to answer this reliably."
4. Keep the answer concise.
5. Mention the source when possible.
"""


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)

    def answer(self, q, results):

        context = "\n\n".join(
            f"[Source: {r.chunk.source} | "
            f"Strategy: {r.chunk.strategy} | "
            f"Score: {r.score:.3f}]\n"
            f"{r.chunk.text}"
            for r in results
        )

        prompt = f"""{SYSTEM}

QUESTION:
{q}

CONTEXT:
{context}
"""

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        answer = response.text.strip()

        return answer, context
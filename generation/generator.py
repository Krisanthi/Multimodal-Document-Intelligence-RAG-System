"""Answer generation via Groq API (Llama 3.3 70B)."""

import logging
from typing import List, Dict, Any

from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document analysis assistant. Answer questions using ONLY the provided context passages.

Rules:
1. Write a clear, well-structured paragraph answer based only on the provided context.
2. Do NOT fabricate information. If the context is insufficient, state that clearly.
3. At the end, list the sources you referenced in this format:
   Sources: [filename, Chunk N], [filename, Chunk N]
4. Keep inline text clean. Do NOT repeat source references within the paragraph body.
5. Be thorough but concise."""


class AnswerGenerator:
    """Generates answers from retrieved chunks using Groq LLM."""

    MODEL = "llama-3.3-70b-versatile"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"answer": "No relevant documents found. Please upload documents first.",
                    "sources": [], "chunks_used": 0, "model": self.MODEL, "usage": {}}

        context_parts = []
        sources = set()
        for i, c in enumerate(chunks):
            context_parts.append(
                f"[Passage {i+1}] Source: {c.get('source','unknown')} | "
                f"Chunk: {c.get('chunk_index',0)} | Type: {c.get('modality','text')}\n{c['text']}")
            sources.add(c.get("source", "unknown"))

        user_msg = (f"Context:\n\n" + "\n\n".join(context_parts) +
                    f"\n\nQuestion: {question}")

        try:
            resp = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE,
                top_p=0.9, stream=False)

            return {
                "answer": resp.choices[0].message.content.strip(),
                "sources": sorted(sources),
                "chunks_used": len(chunks),
                "model": self.MODEL,
                "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                          "completion_tokens": resp.usage.completion_tokens,
                          "total_tokens": resp.usage.total_tokens},
            }
        except Exception as e:
            logger.error("Groq API error: %s", e)
            return {"answer": f"Error generating answer: {e}",
                    "sources": sorted(sources), "chunks_used": len(chunks),
                    "model": self.MODEL, "usage": {}}

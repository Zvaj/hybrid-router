import anthropic
import json
import os

from backends.ckg import CKGBackend
from backends.rag import RAGBackend

DEFAULT_MODEL = "claude-sonnet-4-6"

MODEL_SHORTCUTS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}

CKG_TYPES = {"T2", "T3", "T4"}
RAG_TYPES = {"T1", "T5"}

PAPER_RAG_TOKENS = 2982
PAPER_RAG_F1 = 0.1231
PAPER_CKG_TOKENS = 269
PAPER_CKG_F1 = 0.4709


def format_ckg_context(retrieval):
    return json.dumps(retrieval, indent=2)


def format_rag_context(chunks):
    return "\n\n---\n\n".join(chunks)


def call_llm(query, context, source,
             model=DEFAULT_MODEL):
    if source == "CKG":
        retrieval_type = ""
        try:
            import json as _json
            parsed = _json.loads(context)
            retrieval_type = parsed.get("type", "")
        except Exception:
            pass

        if retrieval_type == "dependents":
            concept = ""
            try:
                concept = _json.loads(context).get("concept", "the concept")
            except Exception:
                pass
            system = (
                "You are a precise tutor using a knowledge graph. "
                f"These concepts build directly on {concept} and are the natural next steps to study. "
                "List them in order from most immediate to most advanced. "
                "Only use concepts present in the graph data."
            )
        else:
            system = (
                "You are a precise tutor using a knowledge graph. "
                "Answer questions about concept relationships and prerequisites "
                "concisely. List concepts in order from most foundational to "
                "most advanced. Only use concepts present in the graph data."
            )
    else:
        system = (
            "You are a helpful calculus tutor. Use the provided "
            "reference material to explain concepts clearly with examples "
            "where helpful. Be accurate and educational."
        )

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # Build kwargs — omit temperature for Opus 4.7
        # which uses adaptive thinking and rejects it
        api_kwargs = {
            "model": model,
            "max_tokens": 500,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        }
        if model != "claude-opus-4-7":
            api_kwargs["temperature"] = 0
        message = client.messages.create(**api_kwargs)
        return message.content[0].text.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def route(query, query_type, ckg_backend, rag_backend,
          dry_run=False, model=DEFAULT_MODEL):
    if query_type in CKG_TYPES:
        retrieval = ckg_backend.retrieve(query, query_type)
        if retrieval is None:
            source = "RAG (CKG fallback)"
            chunks = rag_backend.retrieve(query)
            context = format_rag_context(chunks)
        else:
            source = "CKG"
            context = format_ckg_context(retrieval)
    else:
        source = "RAG"
        chunks = rag_backend.retrieve(query)
        context = format_rag_context(chunks)

    context_tokens = len(context.split())
    tokens_saved = max(0, PAPER_RAG_TOKENS - context_tokens)
    estimated_cost = round(
        (context_tokens * 3 / 1_000_000) + (200 * 15 / 1_000_000),
        6,
    )

    if dry_run:
        answer = (
            f"[DRY RUN — {source} path — "
            f"{context_tokens} tokens — model: {model}]"
        )
    else:
        answer = call_llm(query, context, source, model=model)

    return {
        "query": query,
        "query_type": query_type,
        "source": source,
        "context": context,
        "context_tokens": context_tokens,
        "tokens_saved": tokens_saved,
        "estimated_cost": estimated_cost,
        "answer": answer,
    }

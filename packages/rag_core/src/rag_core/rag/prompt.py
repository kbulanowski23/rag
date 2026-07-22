"""Prompt construction and context budgeting.

Two things matter here and both are about trust:

1. The model must be told to answer *only* from the context and to say so when
   the context does not contain the answer. An enterprise search tool that
   confidently invents a policy number is worse than one that returns nothing.
2. Every context block is numbered, and the model is required to cite those
   numbers. The numbers map back to real chunks with real page references, so a
   user can verify any claim.
"""

from __future__ import annotations

from pathlib import Path

from rag_core.config import RetrievalSettings
from rag_core.documents import SearchHit
from rag_core.llm.base import Message

DEFAULT_SYSTEM_PROMPT = """\
You are an enterprise document search assistant. You answer questions using only \
the numbered source excerpts provided in the user message.

Rules:
- Use only the provided excerpts. Do not use outside knowledge, and do not guess.
- If the excerpts do not contain the answer, say so plainly and state what \
information would be needed. Do not pad the answer with related-but-unasked-for \
material.
- Cite the excerpts you used inline with bracketed numbers, like [1] or [2][3]. \
Place the citation immediately after the claim it supports.
- If excerpts disagree with each other, say so and cite both.
- Quote exact figures, dates, identifiers and names verbatim from the excerpts. \
Never round or paraphrase a number.
- Be concise. Prefer a direct answer over a summary of the documents.
"""

USER_TEMPLATE = """\
Source excerpts:

{context}

---

Question: {question}"""


def load_system_prompt(path: str | None) -> str:
    """Allows the prompt to be overridden from a ConfigMap without a rebuild."""
    if path:
        p = Path(path)
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


def format_excerpt(index: int, hit: SearchHit) -> str:
    header = f"[{index}] {hit.citation_label()}"
    if hit.extraction_source == "ocr":
        # Flagged so the model treats mangled tokens with appropriate suspicion.
        header += " (text recovered by OCR; may contain recognition errors)"
    return f"{header}\n{hit.text.strip()}"


def select_within_budget(
    hits: list[SearchHit], budget_tokens: int, chars_per_token: int = 4
) -> list[SearchHit]:
    """Take hits in rank order until the context budget is spent.

    Approximate on purpose: the exact tokenizer for the target LLM is unknown and
    varies by provider, so we budget conservatively in characters instead of
    pretending to a precision we do not have.
    """
    budget_chars = budget_tokens * chars_per_token
    selected: list[SearchHit] = []
    used = 0
    for hit in hits:
        cost = len(hit.text) + 120  # header and separators
        if selected and used + cost > budget_chars:
            break
        selected.append(hit)
        used += cost
    return selected


def build_messages(
    question: str,
    hits: list[SearchHit],
    settings: RetrievalSettings,
    system_prompt: str | None = None,
    history: list[Message] | None = None,
) -> tuple[list[Message], list[SearchHit]]:
    """Returns the messages to send and the hits actually included, in the same
    order as their citation numbers."""
    used = select_within_budget(hits, settings.context_token_budget)
    if used:
        context = "\n\n".join(format_excerpt(i, h) for i, h in enumerate(used, start=1))
    else:
        context = "(no matching documents were found in the index)"

    messages: list[Message] = [Message("system", system_prompt or DEFAULT_SYSTEM_PROMPT)]
    if history:
        # Prior turns give the model referents for "it" and "that policy".
        # Sources are not re-attached; only the current turn's context is.
        messages.extend(history)
    messages.append(Message("user", USER_TEMPLATE.format(context=context, question=question)))
    return messages, used

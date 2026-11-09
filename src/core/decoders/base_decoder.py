"""Shared model access and constrained token selection."""

from typing import List, Set

from src.core.llm_core import LLM
from src.errors import CallMeMaybeError


class DecoderBase:
    """Provide common operations for specialized decoders."""

    def __init__(self, llm: LLM) -> None:
        """Reuse the model instance supplied by the application."""
        self.llm = llm

    def get_best_token(
        self,
        logits: List[float],
        allowed_token_ids: Set[int]
    ) -> int:
        """Choose the highest-scoring allowed token, resolving ties by ID."""
        if not allowed_token_ids:
            raise CallMeMaybeError("No valid token is available.")

        return max(
            sorted(allowed_token_ids),
            key=lambda token_id: logits[token_id]
        )


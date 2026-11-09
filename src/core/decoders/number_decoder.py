"""Constrained generation of decimal numbers and integers."""

from math import isfinite
import re
from typing import List, Set

from src.core.decoders.base_decoder import DecoderBase
from src.core.llm_core import LLM
from src.core.vocab_processor import VocabProcessor
from src.errors import CallMeMaybeError


class NumberDecoder(DecoderBase):
    """Generate numeric parameters using vocabulary and prefix constraints."""

    def __init__(self, llm: LLM) -> None:
        """Prepare numeric vocabulary while reusing the existing model."""
        super().__init__(llm)
        self.vocab = VocabProcessor(llm)
        self.stop_token_ids: Set[int] = set()
        if self.vocab.end_token_id is not None:
            self.stop_token_ids.add(self.vocab.end_token_id)
        for whitespace in (" ", "\n", "\t", "\r", "\n\n"):
            for token_id in self.llm.encode(whitespace):
                text = self.llm.decode([token_id])
                if text and text.isspace():
                    self.stop_token_ids.add(token_id)

    def get_number_token_ids(
        self,
        prefix: str,
        integer: bool = False,
    ) -> Set[int]:
        """Allow only tokens extending a valid decimal number prefix."""
        pattern = r"-?(?:0|[1-9][0-9]*)"
        if not integer:
            pattern += r"(?:\.[0-9]*)?"

        candidates = set(self.vocab.digit_token_ids)
        for token_id in (
            self.vocab.minus_token_id,
            self.vocab.dot_token_id if not integer else None,
        ):
            if token_id is not None:
                candidates.add(token_id)

        allowed: Set[int] = set()
        for token_id in candidates:
            candidate = prefix + self.vocab.token_by_id[token_id]
            if candidate == "-" or re.fullmatch(pattern, candidate):
                allowed.add(token_id)

        if prefix and prefix[-1].isdigit():
            allowed.update(self.stop_token_ids)

        return allowed

    def generate_numeric_text(
        self,
        prompt_token_ids: List[int],
        integer: bool = False,
        max_tokens: int = 64,
    ) -> str:
        """Generate a decimal value, ending at a separator or whitespace."""
        if (
            not self.vocab.digit_token_ids
            or self.vocab.minus_token_id is None
            or self.vocab.end_token_id is None
            or (not integer and self.vocab.dot_token_id is None)
        ):
            raise CallMeMaybeError("Vocabulary lacks required numeric tokens.")
        generated_tokens: List[int] = []
        prefix = ""

        for _ in range(max_tokens):
            allowed = self.get_number_token_ids(prefix, integer)
            logits = self.llm.get_logits(prompt_token_ids + generated_tokens)
            next_token = self.get_best_token(logits, allowed)

            if next_token in self.stop_token_ids:
                return prefix

            generated_tokens.append(next_token)
            prefix += self.vocab.token_by_id[next_token]

        raise CallMeMaybeError(
            f"LLM did not finish number generation after {max_tokens} tokens. "
            f"Generated prefix: {prefix!r}"
        )

    def generate_number(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
    ) -> float:
        """Generate a finite float suitable for JSON serialization."""
        value = float(self.generate_numeric_text(
            prompt_token_ids, max_tokens=max_tokens,
        ))
        if not isfinite(value):
            raise CallMeMaybeError("Generated number exceeds float range.")
        return value

    def generate_integer(self, prompt_token_ids: List[int]) -> int:
        """Generate an integer directly without rounding a float."""
        return int(self.generate_numeric_text(prompt_token_ids, integer=True))


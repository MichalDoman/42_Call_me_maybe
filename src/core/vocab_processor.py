import json
from typing import Dict, Set

from src.core.llm_core import LLM
from src.errors import CallMeMaybeError


class VocabProcessor:
	def __init__(self, llm: LLM) -> None:
		self.llm = llm

		self.token_by_id: Dict[int, str] = {}
		self.digit_token_ids: Set[int] = set()
		self.minus_token_id: int | None = None
		self.dot_token_id: int | None = None
		self.end_token_id: int | None = None

		self.load_vocab()
		self.extract_tokens()

	def load_vocab(self) -> None:
		try:
			vocab_path = self.llm.model.get_path_to_vocab_file()

			with open(vocab_path, "r", encoding="utf-8") as file:
				vocab: Dict[str, int] = json.load(file)

		except (OSError, json.JSONDecodeError):
			raise CallMeMaybeError("Unable to load vocabulary")

		for token, token_id in vocab.items():
			self.token_by_id[token_id] = token

	def extract_tokens(self) -> None:
		for token_id, token in self.token_by_id.items():
			if token.isascii() and token.isdigit():
				self.digit_token_ids.add(token_id)
			
			if token == "-":
				self.minus_token_id = token_id
			
			if token == ".":
				self.dot_token_id = token_id
			
			if token == "|":
				self.end_token_id = token_id

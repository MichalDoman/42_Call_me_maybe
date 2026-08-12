from typing import List, Set

from src.core.llm_core import LLM
from src.core.models import FunctionDefinition


class ConstrainedDecoder:
	def __init__(self, llm: LLM) -> None:
		self.llm = llm

	def tokenize_functions_names(
		self,
		functions: List[FunctionDefinition],
	) -> List[List[int]]:
		tokenized_functions_names: List[List[int]] = []
		
		for function in functions:
			tokenized_functions_names.append(
				self.llm.encode(function.name)
			)
		
		return tokenized_functions_names

	def get_allowed_tokens(
		self,
		tokenized_functions_names: List[List[int]],
		generated_tokens: List[int]
	) -> Set[int]:
		allowed_tokens: Set[int] = set()
		
		if not generated_tokens:
			for tokens in tokenized_functions_names:
				allowed_tokens.add(tokens[0])
			return allowed_tokens

		for tokens in tokenized_functions_names:
			for token_id, token in enumerate(generated_tokens):
				is_valid: bool = True
				if token != tokens[token_id]:
					is_valid = False

			if not is_valid:
				continue

			if len(tokens) > len(generated_tokens):
				allowed_tokens.add(tokens[len(generated_tokens)])

		return allowed_tokens

	def get_best_token(
		self,
		logits: List[float],
		allowed_tokens: List[int]
	) -> int:
		pass

	def change_logits_values() -> List[float]:
		pass
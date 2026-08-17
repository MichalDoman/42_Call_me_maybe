from typing import List, Set
from math import inf

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
		
		for tokenized_name in tokenized_functions_names:
			tokenized_name.append(91)
		
		return tokenized_functions_names

	def get_allowed_token_ids(
		self,
		tokenized_functions_names: List[List[int]],
		generated_tokens: List[int]
	) -> Set[int]:
		allowed_token_ids: Set[int] = set()

		for tokens in tokenized_functions_names:
			if len(generated_tokens) < len(tokens):
				if tokens[:len(generated_tokens)] == generated_tokens:
					allowed_token_ids.add(
						tokens[len(generated_tokens)]
					)

		return allowed_token_ids

	def get_best_token(
		self,
		logits: List[float],
		allowed_token_ids: List[int]
	) -> int:
		masked_logits: List[int] = len(logits) * [-inf]

		for token_id in allowed_token_ids:
			masked_logits[token_id] = logits[token_id]

		return logits.index(max(masked_logits))

	def generate_function_name(
		self,
		prompt_token_ids: List[int],
		functions: List[FunctionDefinition]
	) -> str:
		generated_tokens: List[int] = []

		tokenized_names = self.tokenize_functions_names(functions)

		while True:
			allowed_token_ids = self.get_allowed_token_ids(
				tokenized_names,
				generated_tokens
			)

			if not allowed_token_ids:
				break

			ids_to_prompt = prompt_token_ids + generated_tokens
			logits = self.llm.get_logits(ids_to_prompt)

			next_token = self.get_best_token(logits, allowed_token_ids)
			generated_tokens.append(next_token)

			if generated_tokens in tokenized_names:
				break

		return self.llm.decode(generated_tokens[:-1])

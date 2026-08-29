from typing import List, Set
from math import inf

from src.core.llm_core import LLM
from src.core.models import FunctionDefinition
from src.errors import CallMeMaybeError


class ConstrainedDecoder:
	def __init__(self, llm: LLM) -> None:
		self.llm = llm

	def tokenize(self, elements: List[str]) -> List[List[int]]:
		return [
			self.llm.encode(element + "|")
			for element in elements
		]

	def get_allowed_token_ids(
		self,
		tokenized_elements: List[List[int]],
		generated_tokens: List[int]
	) -> Set[int]:
		allowed_token_ids: Set[int] = set()

		for tokens in tokenized_elements:
			if len(generated_tokens) >= len(tokens):
				continue

			if tokens[:len(generated_tokens)] == generated_tokens:
				allowed_token_ids.add(tokens[len(generated_tokens)])

		return allowed_token_ids

	def get_best_token(
		self,
		logits: List[float],
		allowed_token_ids: Set[int]
	) -> int:
		masked_logits: List[float] = len(logits) * [-inf]

		for token_id in allowed_token_ids:
			masked_logits[token_id] = logits[token_id]

		return masked_logits.index(max(masked_logits))

	def generate_from_list(
		self,
		prompt_token_ids: List[int],
		elements: List[str],
	) -> str:
		generated_tokens: List[int] = []

		tokenized_elements = self.tokenize(elements)

		while True:
			allowed_token_ids = self.get_allowed_token_ids(
				tokenized_elements,
				generated_tokens
			)

			if not allowed_token_ids:
				break

			input_ids = prompt_token_ids + generated_tokens
			logits = self.llm.get_logits(input_ids)

			next_token = self.get_best_token(
				logits,
				allowed_token_ids
			)

			generated_tokens.append(next_token)

			if generated_tokens in tokenized_elements:
				break

		return self.llm.decode(generated_tokens).removesuffix("|")

	def generate_function_name(
		self,
		prompt_token_ids: List[int],
		functions: List[FunctionDefinition],
	) -> str:
		function_names = [
			function.name
			for function in functions
		]

		return self.generate_from_list(prompt_token_ids, function_names)

	def get_numbers(self, user_input: str, extracted_params: List[str]) -> List[str]:
		numbers: List[str] = []
		current: str = ""

		for char in user_input + " ":
			if char.isdigit():
				current += char

			elif char == "." and current and "." not in current:
				current += char

			elif char == "-" and not current:
				current = "-"

			else:
				if current not in {"", "-"}:
					numbers.append(current)

				current = ""

		return [
			number
			for number in numbers
			if number not in extracted_params
		]

	def generate_integer(self, prompt_token_ids: List[int]) -> int:
		pass

	def generate_string(self, prompt_token_ids: List[int]) -> str:
		pass

	def generate_boolean(self, prompt_token_ids: List[int]) -> bool:
		result = self.generate_from_list(
			prompt_token_ids,
			["true", "false"]
		)
		return result == "true"

	def generate_parameter_value(
		self,
		user_input: str,
		extracted_params: List[str],
		prompt_token_ids: List[int],
		parameter_type: str
	) -> float | int | str | bool:
		if parameter_type == "number":
			numbers = self.get_numbers(user_input, extracted_params)
			return self.generate_from_list(prompt_token_ids, numbers)

		if parameter_type == "integer":
			return self.generate_integer(prompt_token_ids)

		if parameter_type == "string":
			return self.generate_string(prompt_token_ids)

		if parameter_type == "boolean":
			return self.generate_boolean(prompt_token_ids)

		raise CallMeMaybeError(f"Unsupported parameter type: {parameter_type}")

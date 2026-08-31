from typing import List, Set
from math import inf

from src.core.llm_core import LLM
from src.core.models import FunctionDefinition
from src.errors import CallMeMaybeError
from src.core.vocab_processor import VocabProcessor


class ConstrainedDecoder:
	def __init__(self, llm: LLM) -> None:
		self.llm = llm
		self.vocab = VocabProcessor(self.llm)

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

	def generate_raw_value(
		self,
		prompt_token_ids: List[int],
		max_tokens: int = 10,
	) -> str:
		generated_tokens: List[int] = []

		for _ in range(max_tokens):
			input_ids = prompt_token_ids + generated_tokens
			logits = self.llm.get_logits(input_ids)

			next_token = logits.index(max(logits))
			generated_tokens.append(next_token)

			generated_text = self.llm.decode(generated_tokens)

			if "|" in generated_text:
				return generated_text.split("|", 1)[0].strip()

		raise CallMeMaybeError(
			"LLM did not finish parameter generation."
		)

	def generate_number(
		self,
		prompt_token_ids: List[int],
		max_tokens: int = 20,
	) -> float:
		generated_tokens: List[int] = []
		has_digit = False
		has_dot = False

		for _ in range(max_tokens):
			allowed_token_ids: Set[int] = set(
				self.vocab.digit_token_ids
			)

			# Minus tylko na samym początku.
			if not generated_tokens:
				if self.vocab.minus_token_id is not None:
					allowed_token_ids.add(
						self.vocab.minus_token_id
					)

			# Kropka dopiero po co najmniej jednej cyfrze
			# i tylko jedna.
			if has_digit and not has_dot:
				if self.vocab.dot_token_id is not None:
					allowed_token_ids.add(
						self.vocab.dot_token_id
					)

			# Można zakończyć tylko poprawną liczbę.
			# Jeśli ostatnim znakiem jest ".", jeszcze nie kończymy.
			if (
				has_digit
				and generated_tokens
				and generated_tokens[-1] != self.vocab.dot_token_id
			):
				if self.vocab.end_token_id is not None:
					allowed_token_ids.add(
						self.vocab.end_token_id
					)

			input_ids = prompt_token_ids + generated_tokens
			logits = self.llm.get_logits(input_ids)

			next_token = self.get_best_token(
				logits,
				allowed_token_ids,
			)

			generated_tokens.append(next_token)

			if next_token == self.vocab.end_token_id:
				value = self.llm.decode(
					generated_tokens[:-1]
				)

				return float(value)

			if next_token == self.vocab.dot_token_id:
				has_dot = True

			if next_token in self.vocab.digit_token_ids:
				has_digit = True

		raise CallMeMaybeError(
			"LLM did not finish number generation."
		)

	# def generate_number(
	# 	self,
	# 	user_input: str,
	# 	extracted_params: List[int],
	# 	prompt_token_ids: List[int]
	# ) -> int | float:
	# 	numbers = self.get_numbers(user_input, extracted_params)

	# 	if numbers:
	# 		result = self.generate_from_list(prompt_token_ids, numbers)
	# 		return float(result)

	# 	return self.generate_raw_value(prompt_token_ids)

	def generate_string(
		self,
		prompt_token_ids: List[int],
		max_tokens: int = 10,
	) -> str:
		generated_tokens: List[int] = []

		for _ in range(max_tokens):
			input_ids = prompt_token_ids + generated_tokens
			logits = self.llm.get_logits(input_ids)

			next_token = logits.index(max(logits))
			generated_tokens.append(next_token)

			generated_text = self.llm.decode(generated_tokens)

			if "|" in generated_text:
				return generated_text.split("|", 1)[0].strip()

		raise CallMeMaybeError(
			"LLM did not finish string parameter generation."
		)

	def generate_boolean(self, prompt_token_ids: List[int]) -> bool:
		result = self.generate_from_list(
			prompt_token_ids,
			["true", "false"]
		)
		return result == "true"

	def get_words(self, user_input: str) -> List[str]:
		split_user_input = user_input.split()

		for word in split_user_input:
			if (word[0], word[-1]) in [("'", "'"), ("\"", "\"")]:
				return [word[1:-1]]
		
		return split_user_input


	def generate_parameter_value(
		self,
		user_input: str,
		extracted_params: List[str],
		prompt_token_ids: List[int],
		parameter_type: str
	) -> float | int | str | bool:
		if parameter_type == "number":
			return self.generate_number(
				prompt_token_ids
			)

		if parameter_type == "integer":
			return int(self.generate_number(
				user_input,
				extracted_params,
				prompt_token_ids
			))

		if parameter_type == "string":
			return self.generate_from_list(
				prompt_token_ids,
				self.get_words(user_input)
			)

		if parameter_type == "boolean":
			return self.generate_boolean(prompt_token_ids)

		raise CallMeMaybeError(f"Unsupported parameter type: {parameter_type}")

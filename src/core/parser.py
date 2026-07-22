import json
from typing import List
from pathlib import Path
from pydantic import ValidationError

from src.core.models import (
	FunctionDefinition,
	PromptDefinition,
	ModelResponseDefinition,
)
from src.settings import FUNCTIONS_PATH, PROMPTS_PATH
from src.errors import ParsingError


class Parser:
	@staticmethod
	def read_json(path: Path) -> object:
		try:
			with path.open(encoding="utf-8") as file:
				return json.load(file)
		except FileNotFoundError:
			raise ParsingError(
				f"File {path} does not exist."
			)
		except json.JSONDecodeError:
			raise ParsingError(
				f"Invalid JSON format in file: {path}"
			)

	@staticmethod
	def load_function_definitions() -> List[FunctionDefinition]:
		functions: List[FunctionDefinition] = []

		data = Parser.read_json(FUNCTIONS_PATH)
		if not isinstance(data, list):
			raise ParsingError("Invalid JSON type format. Has to be a list.")

		try:
			for item in data:
				function = FunctionDefinition.model_validate(item)
				functions.append(function)
		except ValidationError:
			raise ParsingError(f"Invalid parameter at: {item}")

		return functions

	@staticmethod
	def load_prompts() -> List[PromptDefinition]:
		prompts: List[PromptDefinition] = []

		data = Parser.read_json(PROMPTS_PATH)
		if not isinstance(data, list):
			raise ParsingError("Invalid JSON type format. Has to be a list.")
		try:
			for item in data:
				prompt = PromptDefinition.model_validate(item)
				prompts.append(prompt)
		except ValidationError:
			raise ParsingError(f"Invalid parameter at: {item}")

		return prompts

	@staticmethod
	def parse_model_output(output: str) -> ModelResponseDefinition:
		try:
			data = json.loads(output)
		except json.JSONDecodeError:
			raise ParsingError("Model did not return valid JSON file")

		try:
			result = ModelResponseDefinition.model_validate(data)
		except ValidationError:
			raise ParsingError("Invalid model output definition.")
		
		return result

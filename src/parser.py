import json
from typing import List
from pathlib import Path

from src.settings import FUNCTIONS_PATH, PROMPTS_PATH
from src.errors import ParsingError


class Parser():
	def read_json(self, path: Path) -> None:
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

	def load_functions(self) -> None:
		data = self.read_json(FUNCTIONS_PATH)
		return data

	def load_prompts(self):
		return []

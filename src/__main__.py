from pathlib import Path

from src.parser import Parser
from src.errors import CallMeMaybeError


def debug_reading_json(prompts, functions) -> None:
	print(f"Loaded {len(functions)} function definitions.")
	print(f"Loaded {len(prompts)} prompts.")
	
	for function in functions:
		print(f"- {function}")

	for prompt in prompts:
		print(f"- {prompt}")


def main() -> None:
	try:
		prompts = Parser.load_prompts()
		functions = Parser.load_functions()
		debug_reading_json(prompts, functions)
	except CallMeMaybeError as e:
		print(e)


if __name__ == "__main__":
	main()

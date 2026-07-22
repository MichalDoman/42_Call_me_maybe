from pathlib import Path

from src.core.prompt import prompt_build
from src.core.parser import Parser
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
		functions = Parser.load_function_definitions()
		# debug_reading_json(prompts, functions)
		print(prompt_build(prompts[0], functions))
	except CallMeMaybeError as e:
		print(e)


if __name__ == "__main__":
	main()

from pathlib import Path

from src.core.prompt import prompt_build
from src.core.parser import Parser
from src.core.llm_core import LLM
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
		prompt = prompt_build(prompts[0], functions)
		llm = LLM()

		token_ids = llm.encode("What is the sum of 2 and 3?")
		logits = llm.get_logits(token_ids)

		print("Token IDs:")
		print(token_ids)
        
		print("Number of possible tokens:")
		print(len(logits))

	except CallMeMaybeError as e:
		print(e)


if __name__ == "__main__":
	main()

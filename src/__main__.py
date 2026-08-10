from pathlib import Path

from src.core.prompt import prompt_build
from src.core.parser import Parser
from src.core.llm_core import LLM
from src.core.constrained_decoder import ConstrainedDecoder
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
		constrained_decoder = ConstrainedDecoder(llm)

		token_ids = llm.encode(prompt)
		logits = llm.get_logits(token_ids)
		tokenized_names = constrained_decoder.tokenize_functions_names(functions)

		constrained_decoder.get_allowed_tokens(tokenized_names, [8822, 1889, 3744])

	except CallMeMaybeError as e:
		print(e)


if __name__ == "__main__":
	main()

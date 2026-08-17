from pathlib import Path

from src.core.prompt import prompt_build
from src.core.parser import Parser
from src.core.llm_core import LLM
from src.core.constrained_decoder import ConstrainedDecoder
from src.errors import CallMeMaybeError


def main() -> None:
	try:
		prompts = Parser.load_prompts()
		functions = Parser.load_function_definitions()
		prompt = prompt_build("Add water and sand?", functions)
		llm = LLM()
		constrained_decoder = ConstrainedDecoder(llm)

		prompt_token_ids = llm.encode(prompt)
		generated_function_name = constrained_decoder.generate_function_name(
			prompt_token_ids,
			functions
		)
		print(generated_function_name)

	except CallMeMaybeError as e:
		print(e)


if __name__ == "__main__":
	main()

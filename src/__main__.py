from pathlib import Path

from src.core.prompt import prompt_function_name_build, prompt_parameters_build
from src.core.parser import Parser
from src.core.llm_core import LLM
from src.core.constrained_decoder import ConstrainedDecoder
from src.errors import CallMeMaybeError


def main() -> None:
	try:
		prompts = Parser.load_prompts()
		functions = Parser.load_function_definitions()
		llm = LLM()
		decoder = ConstrainedDecoder(llm)

		for user_input in prompts[:1]:
			system_prompt = prompt_function_name_build(user_input.prompt, functions)
			prompt_token_ids = llm.encode(system_prompt)
			generated_function_name = decoder.generate_function_name(
				prompt_token_ids,
				functions
			)
			# print(generated_function_name)

			function = [
				f for f in functions 
				if f.has_given_name(generated_function_name)
			][0]

			generated_parameters: Dict[str, Any] = {}
			for parameter, definition in function.parameters.items():
				system_prompt = prompt_parameters_build(
					user_input.prompt,
					parameter,
					function
				)
				prompt_token_ids = llm.encode(system_prompt)
				value = decoder.generate_parameter_value(
					prompt_token_ids,
					definition.type
				)
				generated_parameters[parameter] = value
			
			print(generated_parameters)

	except CallMeMaybeError as e:
		print(e)


if __name__ == "__main__":
	main()

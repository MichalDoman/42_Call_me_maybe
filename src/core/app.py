from pathlib import Path

from src.core.prompt import prompt_function_name_build, prompt_parameters_build
from src.core.parser import Parser
from src.core.llm_core import LLM
from src.core.constrained_decoder import ConstrainedDecoder
from src.errors import CallMeMaybeError


class App:
	def __init__(self) -> None:
		self.prompts = Parser.load_prompts()
		self.functions = Parser.load_function_definitions()
		self.llm = LLM()

		self.decoder = ConstrainedDecoder(self.llm)

	def run(self) -> None:
		try:
			for user_input in self.prompts[:1]:
				system_prompt = prompt_function_name_build(user_input.prompt, self.functions)
				prompt_token_ids = self.llm.encode(system_prompt)
				generated_function_name = self.decoder.generate_function_name(
					prompt_token_ids,
					self.functions
				)
				print(generated_function_name)

				function = [
					f for f in self.functions 
					if f.has_given_name(generated_function_name)
				][0]

				generated_parameters: Dict[str, Any] = {}
				for parameter, definition in function.parameters.items():
					system_prompt = prompt_parameters_build(
						user_input.prompt,
						parameter,
						definition.type,
						function
					)
					prompt_token_ids = self.llm.encode(system_prompt)
					value = self.decoder.generate_parameter_value(
						user_input.prompt,
						[str(value) for _, value in generated_parameters.items()],
						prompt_token_ids,
						definition.type
					)
					generated_parameters[parameter] = value
				
				print(generated_parameters)

		except CallMeMaybeError as e:
			print(e)

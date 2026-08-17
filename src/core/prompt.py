import json
from typing import List, Dict

from src.core.models import FunctionDefinition

def prompt_build(
	user_input: str,
	functions: List[FunctionDefinition]
) -> str:
	dumped_functions: List[Dict[str, object]] = []

	for function in functions:
		dumped_function = function.model_dump()
		dumped_functions.append(dumped_function)
	
	json_functions = json.dumps(dumped_functions, indent=2)

	return (
		"Choose exactly one function that best matches the user request and return its name.\n"
		"If the function name is complete, add '|' sign at the end e.g. 'fn_greet|'."
		"Use only a function from the provided list.\n"
		f"Available functions:\n{json_functions}\n\n"
		f"User request:\n{user_input}\n\n"
		"Already generated tokens:\n"
	)

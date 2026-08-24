import json
from typing import List, Dict

from src.core.models import FunctionDefinition

def prompt_function_name_build(
	user_input: str,
	functions: List[FunctionDefinition]
) -> str:
	dumped_functions: List[Dict[str, object]] = []

	for function in functions:
		dumped_function = function.model_dump()
		dumped_functions.append(dumped_function)
	
	json_functions = json.dumps(dumped_functions, indent=2)

	return (
		"Choose exactly one function that best matches the user request.\n"
		"Use only a function from the provided list.\n"
		"Return only the selected function name followed immediately by '|'.\n"
		"The '|' character marks the end of the function name.\n"
		"Do not return explanations, spaces, or any additional text.\n"
		"Output format: function_name|\n\n"
		f"Available functions:\n{json_functions}\n\n"
		f"User request:\n{user_input}\n\n"
		"Selected function:\n"
	)

def prompt_parameters_build(
	user_input: str,
	parameter_name: str,
	function: FunctionDefinition
) -> str:
	dumped_function = function.model_dump()

	return (
        "Extract the value of the given parameter, required by "
		"the selected function from the user request.\n"
		"Return only the selected parameter value followed immediately by '|'.\n"
		"The '|' character marks the end of the parameter value.\n"
		"Output format: parameter_value|\n\n"
		f"User request:\n{user_input}\n\n"
        f"Parameter name:\n{parameter_name}\n\n"
		f"Function definition:\n{dumped_function}\n"
        "Already generated tokens:\n"
    )
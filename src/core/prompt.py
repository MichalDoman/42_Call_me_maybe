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
    parameter_type: str,
    function: FunctionDefinition,
) -> str:
    json_function = json.dumps(
        function.model_dump(),
        indent=2,
    )

    return (
        "Extract exactly one parameter value from the user request.\n"
        "Use the function description and parameter definitions "
        "to determine which value belongs to the requested parameter.\n"
        f"The required parameter is '{parameter_name}' "
        f"and its type is '{parameter_type}'.\n"
        "Return only its value followed immediately by '|'.\n"
        "Do not explain your answer and do not add extra text.\n"
        "For number or integer parameters, return only a numeric value.\n"
        "For boolean parameters, return only true or false.\n"
        "For string parameters, return only the string value.\n"
        "The '|' character marks the end of the value.\n\n"
        f"Function definition:\n{json_function}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Value for '{parameter_name}':\n"
    )
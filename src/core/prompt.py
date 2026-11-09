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

    type_instructions = {
        "number": (
            "The parameter type is number.\n"
            "Return only a numeric value.\n"
            "The value may be an integer or a decimal number.\n"
            "Convert written numbers to digits when necessary.\n"
            "Do not add quotes or any other text.\n"
        ),
        "integer": (
            "The parameter type is integer.\n"
            "Return only a whole number without a decimal point.\n"
            "Convert written numbers to digits when necessary.\n"
            "Do not add quotes or any other text.\n"
        ),
        "string": (
            "The parameter type is string.\n"
            "Return only the exact text value required by the parameter.\n"
            "Do not add quotes, labels, explanations, or extra words.\n"
            "If the request asks to replace, reverse, transform, or otherwise process a string, "
            "the string to process will usually be enclosed in single quotes '...' "
            'or double quotes "...".\n'
            "In that case, return only the text inside the quotes.\n"
            "Example: for 'Greet shrek' and parameter 'name', return shrek|\n"
            "Example: for \"Reverse the string 'hello'\", return hello|\n"
        ),
        "boolean": (
            "The parameter type is boolean.\n"
            "Return only true| or false|.\n"
            "Infer the boolean value from the meaning of the user request.\n"
            "Do not return yes, no, 1, 0, or any explanation.\n"
        ),
    }

    if parameter_type not in type_instructions:
        type_instruction = (
            f"The parameter type is {parameter_type}.\n"
            "Return only the value followed immediately by '|'.\n"
        )
    else:
        type_instruction = type_instructions[parameter_type]

    return (
        "Extract exactly one value for the specified function parameter.\n"
        "Extract the argument; do not compute the function result.\n"
        "Preserve numeric values exactly; do not append extra digits.\n"
        "Place | immediately after the complete value.\n"
        "Use the meaning of the user request and the function definition "
        "to determine the correct value.\n"
        "Return only the parameter value followed immediately by '|'.\n"
        "The '|' character marks the end of the value.\n"
        "Do not return explanations or additional text.\n\n"
        f"{type_instruction}\n"
        f"Function definition:\n{json_function}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Parameter name: {parameter_name}\n"
        f"Parameter type: {parameter_type}\n\n"
        f"Value for {parameter_name}: "
    )

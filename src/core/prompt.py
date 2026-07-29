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
        "Select exactly one function that best matches the user request.\n"
        "Use only a function from the provided list.\n"
        "Return one valid JSON object with fields 'name' and 'parameters'.\n"
        "Do not include explanations, Markdown, or code blocks.\n\n"
        f"Available functions:\n{json_functions}\n\n"
        f"User request:\n{user_input}\n\n"
        "JSON result:\n"
    )

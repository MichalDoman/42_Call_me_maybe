from typing import Dict

from pydantic import BaseModel


class TypeDefinition(BaseModel):
	"""Describe parameter's type"""

	type: str


class FunctionDefinition(BaseModel):
	"""Describe a function that LLM can use"""

	name: str
	description: str
	parameters: Dict[str, TypeDefinition]
	returns: TypeDefinition


class PromptDefinition(BaseModel):
	"""Describe prompt"""

	prompt: str


class ModelOutputDefinition(BaseModel):
	"""Describe the output of the model"""

	prompt: str
	name: str
	paramenters: Dict[str, TypeDefinition]

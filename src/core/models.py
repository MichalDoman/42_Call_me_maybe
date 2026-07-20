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

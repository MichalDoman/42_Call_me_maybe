from typing import Dict, Any

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


class ModelResponseDefinition(BaseModel):
	"""Describe the response of the model"""

	model_config = ConfigDict(extra="forbid")

	name: str
	parameters: Dict[str, Any]


class FunctionCallResult(ModelResponseDefinition):
	"""Describe the output of the model"""

	prompt: str

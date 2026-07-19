from typing import Dict

from pydantic import BaseModel

class Function(BaseModel):
	"""Describe a function that LLM can use"""

	name: str
	description: str
	parameters: Dict[str, str]
	returns: str

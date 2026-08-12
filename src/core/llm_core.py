from typing import List, Optional

from llm_sdk.llm_sdk import Small_LLM_Model


class LLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model = Small_LLM_Model(model_name)
	
    def encode(self, prompt: str) -> List[int]:
        encoded = self.model.encode(prompt)

        token_ids = encoded.tolist()

        if token_ids and isinstance(token_ids[0], list):
            return token_ids[0]

        return token_ids

    def get_logits(self, token_ids: List[int]) -> List[float]:
        return self.model.get_logits_from_input_ids(token_ids)

    def decode(self, token_ids: List[int]) -> str | None:
        return self.model.decode(token_ids)

    def is_valid_function_name(name: str) -> bool:
        pass
    
    def generate_function_names(
        self,
        prompt_token_ids: List[int],
        functions: List[FunctionDefinition]
    ) -> List[int]:
        pass

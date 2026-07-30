from typing import List

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
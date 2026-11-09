from typing import List, Set

from src.core.llm_core import LLM
from src.core.models import FunctionDefinition
from src.errors import CallMeMaybeError
from src.core.decoders.base_decoder import DecoderBase
from src.core.decoders.number_decoder import NumberDecoder


class ConstrainedDecoder(DecoderBase):
    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        self.number_decoder = NumberDecoder(llm)

    def tokenize(self, elements: List[str]) -> List[List[int]]:
        return [
            self.llm.encode(element + "|")
            for element in elements
        ]

    def get_allowed_token_ids(
        self,
        tokenized_elements: List[List[int]],
        generated_tokens: List[int]
    ) -> Set[int]:
        allowed_token_ids: Set[int] = set()

        for tokens in tokenized_elements:
            if len(generated_tokens) >= len(tokens):
                continue

            if tokens[:len(generated_tokens)] == generated_tokens:
                allowed_token_ids.add(tokens[len(generated_tokens)])

        return allowed_token_ids

    def generate_from_list(
        self,
        prompt_token_ids: List[int],
        elements: List[str],
    ) -> str:
        generated_tokens: List[int] = []

        tokenized_elements = self.tokenize(elements)

        while True:
            allowed_token_ids = self.get_allowed_token_ids(
                tokenized_elements,
                generated_tokens
            )

            if not allowed_token_ids:
                break

            input_ids = prompt_token_ids + generated_tokens
            logits = self.llm.get_logits(input_ids)

            next_token = self.get_best_token(
                logits,
                allowed_token_ids
            )

            generated_tokens.append(next_token)

            if generated_tokens in tokenized_elements:
                break

        return self.llm.decode(generated_tokens).removesuffix("|")

    def generate_function_name(
        self,
        prompt_token_ids: List[int],
        functions: List[FunctionDefinition],
    ) -> str:
        function_names = [
            function.name
            for function in functions
        ]

        return self.generate_from_list(prompt_token_ids, function_names)

    def generate_boolean(self, prompt_token_ids: List[int]) -> bool:
        result = self.generate_from_list(
            prompt_token_ids,
            ["true", "false"]
        )
        return result == "true"

    def get_words(self, user_input: str) -> List[str]:
        split_user_input = user_input.split()

        for word in split_user_input:
            if (word[0], word[-1]) in [("'", "'"), ("\"", "\"")]:
                return [word[1:-1]]

        return split_user_input

    def generate_parameter_value(
        self,
        user_input: str,
        prompt_token_ids: List[int],
        parameter_type: str
    ) -> float | int | str | bool:
        if parameter_type == "number":
            return self.number_decoder.generate_number(
                prompt_token_ids
            )

        if parameter_type == "integer":
            return self.number_decoder.generate_integer(prompt_token_ids)

        if parameter_type == "string":
            return self.generate_from_list(
                prompt_token_ids,
                self.get_words(user_input)
            )

        if parameter_type == "boolean":
            return self.generate_boolean(prompt_token_ids)

        raise CallMeMaybeError(f"Unsupported parameter type: {parameter_type}")

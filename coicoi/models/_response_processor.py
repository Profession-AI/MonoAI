from typing import Dict
from ..tokens.token_counter import TokenCounter
from ..tokens.token_cost import TokenCost
from ..prompts.prompt import Prompt
from pydantic_ai.messages import ModelMessage

class ResponseProcessorMixin:
    def _process_response(
        self,
        prompt: Prompt,
        response: str,
        provider: str,
        model: str,
        count_tokens: bool,
        count_cost: bool
    ) -> Dict:
        """
        Process the response and add optional token and cost information.
        
        Args:
            question: The input question
            answer: The model's answer
            provider: Name of the provider
            model: Name of the model
            count_tokens: Whether to count tokens
            count_cost: Whether to calculate costs
            
        Returns:
            Dictionary containing the response and optional stats
        """

        processed_response = {
            "prompt": str(prompt), 
            "response": response,
            #"messages_trace": response.all_messages(),
            "model": {
                "provider": provider, 
                "name": model
            }
        }
        
        if count_tokens or count_cost:
            tokens = None
            if count_tokens:
                tokens = TokenCounter().count(model, str(prompt), response)
                processed_response["tokens"] = tokens
                
            if count_cost and tokens:
                cost = TokenCost().compute(
                    provider, 
                    model, 
                    tokens["input_tokens"], 
                    tokens["output_tokens"]
                )
                processed_response["cost"] = cost
                
        return processed_response 
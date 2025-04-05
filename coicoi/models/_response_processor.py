from typing import Dict, Optional
from tokens.token_counter import TokenCounter
from tokens.token_cost import TokenCost
from prompts.prompt import Prompt

class ResponseProcessorMixin:
    def _process_response(
        self,
        prompt: Prompt,
        response: str,
        provider_name: str,
        model_name: str,
        count_tokens: bool,
        count_cost: bool
    ) -> Dict:
        """
        Process the response and add optional token and cost information.
        
        Args:
            question: The input question
            answer: The model's answer
            provider_name: Name of the provider
            model_name: Name of the model
            count_tokens: Whether to count tokens
            count_cost: Whether to calculate costs
            
        Returns:
            Dictionary containing the response and optional stats
        """

        processed_response = {
            "prompt": str(prompt), 
            "response": response, 
            "model": {
                "provider": provider_name, 
                "name": model_name
            }
        }
        
        if count_tokens or count_cost:
            tokens = None
            if count_tokens:
                tokens = TokenCounter().count(model_name, str(prompt), response)
                processed_response["tokens"] = tokens
                
            if count_cost and tokens:
                cost = TokenCost().compute(
                    provider_name, 
                    model_name, 
                    tokens["input_tokens"], 
                    tokens["output_tokens"]
                )
                processed_response["cost"] = cost
                
        return processed_response 
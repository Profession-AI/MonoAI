import json
from typing import Dict
from ..tokens.token_counter import TokenCounter
from ..tokens.token_cost import TokenCost
from ..prompts.prompt import Prompt

class ResponseProcessorMixin:
    def _process_response(
        self,
        prompt: Prompt,
        response: Dict
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
        
        if isinstance(prompt, Prompt):
            if prompt.response_type != None:
                response = json.loads(response)["response"]

        processed_response = {
            "prompt": str(prompt), 
            "response": response["choices"][0]["message"]["content"],
            #"messages_trace": response.all_messages(),
            "model": {
                "provider": self.provider, 
                "name": self.model
            }
        }

        if self._count_tokens:
            processed_response["usage"]={
                "completion_tokens":response["usage"]["completion_tokens"],
                "prompt_tokens":response["usage"]["prompt_tokens"],
                "total_tokens=": response["usage"]["total_tokens"]
                }
                        
        return processed_response 


    def _process_chunk(self, chunk):
        delta = chunk["choices"][0]["delta"]["content"]
        return {"delta":delta}
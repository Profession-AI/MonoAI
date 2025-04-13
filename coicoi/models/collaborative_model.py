from ._base_model import BaseModel
from .model import Model
from .multi_model import MultiModel
from ._prompt_executor import PromptExecutorMixin
from ._response_processor import ResponseProcessorMixin
from typing import List, Dict, Union
import asyncio
from ..prompts.prompt_chain import PromptChain
from ..prompts.prompt import Prompt

class CollaborativeModel(BaseModel, PromptExecutorMixin, ResponseProcessorMixin):
    def __init__(
        self,
        models: List[Dict[str, str]],
        aggregator: Dict[str, str],
        count_tokens: bool = False,
        count_cost: bool = False
    ):
        """
        Initialize CollaborativeModel with a list of models and an aggregator model.
        
        Args:
            models: List of dictionaries containing provider and model information
            aggregator: Dictionary containing aggregator model provider and model information
            count_tokens: Whether to count tokens for each request
            count_cost: Whether to calculate costs for each request
        """
        super().__init__(count_tokens, count_cost)

        self._multi_model = MultiModel(
            models=models,
            count_tokens=count_tokens,
            count_cost=count_cost
        )

        self._aggregator = Model(
            aggregator['provider'],
            aggregator['model'],
            count_tokens=count_tokens,
            count_cost=count_cost
        )

    def _format_aggregator_prompt(self, prompt: Union[str, Prompt, PromptChain], responses: List[Dict]) -> str:
        """
        Format the prompt for the aggregator model.
        
        Args:
            prompt: The original prompt or prompt chain
            responses: List of responses from individual models
            
        Returns:
            Formatted prompt for the aggregator
        """
        prompt_text = str(prompt)
        model_responses = "\n\n".join([
            f"Model {i+1} ({response["model"]['provider']} - {response['model']["name"]}):\n{response['response']}"
            for i, response in enumerate(responses)
        ])
        
        return f"""Please analyze the following responses from different models and provide a comprehensive answer:
                    Original Question: {prompt_text}
                    Model Responses:
                    {model_responses}
                    Please provide a well-reasoned response that takes into account all the information above."""


    async def ask_async(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Ask all models and aggregate their responses asynchronously.
        
        Args:
            prompt: The prompt or prompt chain to process
            
        Returns:
            Dictionary containing aggregated response and statistics
        """
        # Get responses from all models
        model_responses = await self._multi_model.ask_async(prompt)
        
        # Get aggregator response
        aggregator_prompt = self._format_aggregator_prompt(prompt, model_responses)
        aggregator_response = await self._execute_async(aggregator_prompt, self._aggregator._agent)
        
        # Process aggregator response
        processed_aggregator = self._process_response(
            aggregator_prompt,
            aggregator_response,
            self._aggregator.provider,
            self._aggregator.model,
            self._count_tokens,
            self._count_cost
        )

        processed_aggregator["individual_responses"] = model_responses
        return processed_aggregator

    def ask(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Ask all models and aggregate their responses synchronously.
        
        Args:
            prompt: The prompt or prompt chain to process
            
        Returns:
            Dictionary containing aggregated response and statistics
        """
        return asyncio.run(self.ask_async(prompt))


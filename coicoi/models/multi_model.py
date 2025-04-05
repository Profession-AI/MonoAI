from ._base_model import BaseModel
from .model import Model
from ._prompt_executor import PromptExecutorMixin
from ._response_processor import ResponseProcessorMixin
from typing import List, Dict, Union
from typing import override
import asyncio
from ..prompts.prompt_chain import PromptChain
from ..prompts.prompt import Prompt

class MultiModel(BaseModel, PromptExecutorMixin, ResponseProcessorMixin):
    def __init__(
        self, 
        models: List[Dict[str, str]], 
        count_tokens: bool = False, 
        count_cost: bool = False
    ):
        """
        Initialize MultiModel with a list of models and counting preferences.
        
        Args:
            models: List of dictionaries containing provider and model information
            count_tokens: Whether to count tokens for each request
            count_cost: Whether to calculate costs for each request
        """
        super().__init__(count_tokens, count_cost)
        self._models = [
            Model(
                model['provider'],
                model['model'],
                count_tokens=count_tokens,
                count_cost=count_cost
            ) for model in models
        ]

    async def _task(self, model: Model, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Execute a single model task.
        
        Args:
            model: The model to use
            prompt: The prompt or prompt chain to process
            
        Returns:
            Dictionary containing the response and optional stats
        """
        response = await self._execute_async(prompt, model._agent)
        return self._process_response(
            prompt,
            response,
            model.provider_name,
            model.model_name,
            self._count_tokens,
            self._count_cost
        )

    @override
    async def ask_async(self, prompt: Union[str, Prompt, PromptChain]) -> List[Dict]:
        """
        Ask all models asynchronously.
        
        Args:
            prompt: The prompt or prompt chain to process
            
        Returns:
            List of dictionaries containing responses and optional stats
        """
        tasks = [self._task(model, prompt) for model in self._models]
        return await asyncio.gather(*tasks)

    @override
    def ask(self, prompt: Union[str, Prompt, PromptChain]) -> List[Dict]:
        """
        Ask all models synchronously.
        
        Args:
            prompt: The prompt or prompt chain to process
            
        Returns:
            List of dictionaries containing responses and optional stats
        """
        return asyncio.run(self.ask_async(prompt))


if __name__ == "__main__":
    # Initialize with counting preferences
    multi_model = MultiModel(
        models=[
            {'provider': 'openai', 'model': 'gpt-4o-mini'},
            {'provider': 'deepseek', 'model': 'deepseek-chat'},
        ],
        count_tokens=True,
        count_cost=True
    )
    
    # Example with simple prompt
    results = multi_model.ask("What is the capital of France?")
    print("\nSimple prompt results:")
    for i, result in enumerate(results, 1):
        print(f"\nModel {i} Response:")
        print(result["output"])
        if "tokens" in result:
            print("Token Count:")
            print(f"Input tokens: {result['tokens']['input_tokens']:,}")
            print(f"Output tokens: {result['tokens']['output_tokens']:,}")
            print(f"Total tokens: {result['tokens']['total_tokens']:,}")
        if "cost" in result:
            print("Cost:")
            print(f"Input cost: ${result['cost']['input_cost']:.6f}")
            print(f"Output cost: ${result['cost']['output_cost']:.6f}")
            print(f"Total cost: ${result['cost']['total_cost']:.6f}")
    
    # Example with prompt chain
    chain = PromptChain([
        "What is the capital of France?",
        "Based on the previous answer, what is its population?"
    ])
    results = multi_model.ask(chain)
    print("\nPrompt chain results:")
    for i, result in enumerate(results, 1):
        print(f"\nModel {i} Response:")
        print(result["output"])
        if "tokens" in result:
            print("Token Count:")
            print(f"Input tokens: {result['tokens']['input_tokens']:,}")
            print(f"Output tokens: {result['tokens']['output_tokens']:,}")
            print(f"Total tokens: {result['tokens']['total_tokens']:,}")
        if "cost" in result:
            print("Cost:")
            print(f"Input cost: ${result['cost']['input_cost']:.6f}")
            print(f"Output cost: ${result['cost']['output_cost']:.6f}")
            print(f"Total cost: ${result['cost']['total_cost']:.6f}")

from model import Model
import asyncio
from typing import List, Dict, Tuple, Optional


class MultiModel:


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
        self._models = [Model(model['provider'], model['model']) for model in models]
        self._count_tokens = count_tokens
        self._count_cost = count_cost


    async def _task(self, model: Model, question: str) -> Tuple[str, Optional[Dict]]:
        """
        Execute a single model task.
        
        Args:
            model: The model to use
            question: The question to ask
            
        Returns:
            Tuple of (response, stats) where stats is None if counting is disabled
        """
        response = await model.ask_async(question)
        stats = None
        
        if self._count_tokens or self._count_cost:
            stats = {}
            if self._count_tokens:
                stats['tokens'] = model.count_tokens(question, response)
            if self._count_cost:
                stats['cost'] = model.count_cost(question, response)
                
        return response, stats


    async def _ask_async(self, question: str) -> List[Tuple[str, Optional[Dict]]]:
        """
        Ask all models asynchronously.
        
        Args:
            question: The question to ask
            
        Returns:
            List of tuples containing (response, stats) for each model
        """
        tasks = [self._task(model, question) for model in self._models]
        return await asyncio.gather(*tasks)


    def ask(self, question: str) -> List[Tuple[str, Optional[Dict]]]:
        """
        Ask all models synchronously.
        
        Args:
            question: The question to ask
            
        Returns:
            List of tuples containing (response, stats) for each model
        """
        return asyncio.run(self._ask_async(question))



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

    # Get responses and stats
    results = multi_model.ask("What is the meaning of life?")
    
    # Print results
    for i, (response, stats) in enumerate(results, 1):
        print(f"\nModel {i} Response:")
        print(response)
        if stats:
            print("Stats:")
            if 'tokens' in stats:
                print(f"Tokens: {stats['tokens']}")
            if 'cost' in stats:
                print(f"Cost: ${stats['cost']}")

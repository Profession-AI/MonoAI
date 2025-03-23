from multi_model import MultiModel
from model import Model
from typing import List, Dict, Optional


class CollectiveModel:
    def __init__(
        self, 
        models: List[Dict[str, str]], 
        aggregator: Dict[str, str],
        count_tokens: bool = False,
        count_cost: bool = False,
        aggregator_prompt: Optional[str] = None
    ):
        """
        Initialize CollectiveModel with models, aggregator, and counting preferences.
        
        Args:
            models: List of dictionaries containing provider and model information
            aggregator: Dictionary containing aggregator provider and model information
            count_tokens: Whether to count tokens for each request
            count_cost: Whether to calculate costs for each request
            aggregator_prompt: Optional custom prompt for the aggregator model
        """
        self._multi_model = MultiModel(
            models=models,
            count_tokens=count_tokens,
            count_cost=count_cost
        )
        self._aggregator = Model(
            provider_name=aggregator['provider'],
            model_name=aggregator['model'],
            count_tokens=count_tokens,
            count_cost=count_cost
        )
        self._aggregator_prompt = aggregator_prompt or "Please aggregate the following answers into a single answer: {answers}"

    def ask(self, question: str) -> Dict:
        """
        Ask all models and aggregate their responses.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the aggregated response and optional stats
        """
        # Get responses from all models
        responses = self._multi_model.ask(question)
        
        # Extract answers and stats
        answers = [response['output'] for response in responses]
        model_stats = [response for response in responses if 'tokens' in response or 'cost' in response]
        
        # Get aggregated response
        aggregated = self._aggregator.ask(
            self._aggregator_prompt.format(answers=answers)
        )
        
        # Combine results
        result = {
            "question": question,
            "aggregated_response": aggregated["output"],
            "model_responses": responses,
            "aggregator": {
                "provider": self._aggregator.provider_name,
                "model": self._aggregator.model_name
            }
        }
        
        # Add stats if available
        if model_stats:
            result["model_stats"] = model_stats
        if "tokens" in aggregated:
            result["aggregator_tokens"] = aggregated["tokens"]
        if "cost" in aggregated:
            result["aggregator_cost"] = aggregated["cost"]
            
        return result


if __name__ == "__main__":
    # Initialize with counting preferences
    collective_model = CollectiveModel(
        models=[
            {'provider': 'openai', 'model': 'gpt-4o-mini'},
            {'provider': 'deepseek', 'model': 'deepseek-chat'},
        ],
        aggregator={'provider': 'deepseek', 'model': 'deepseek-chat'},
        count_tokens=True,
        count_cost=True
    )

    # Get aggregated response with stats
    result = collective_model.ask("What is the capital of France?")
    
    # Print results
    print("\nAggregated Response:")
    print(result["aggregated_response"])
    
    if "model_stats" in result:
        print("\nModel Statistics:")
        for i, stats in enumerate(result["model_stats"], 1):
            print(f"\nModel {i}:")
            if "tokens" in stats:
                print(f"Tokens: {stats['tokens']}")
            if "cost" in stats:
                print(f"Cost: ${stats['cost']}")
    
    if "aggregator_tokens" in result:
        print("\nAggregator Token Count:")
        print(f"Input tokens: {result['aggregator_tokens']['input_tokens']:,}")
        print(f"Output tokens: {result['aggregator_tokens']['output_tokens']:,}")
        print(f"Total tokens: {result['aggregator_tokens']['total_tokens']:,}")
    
    if "aggregator_cost" in result:
        print("\nAggregator Cost:")
        print(f"Input cost: ${result['aggregator_cost']['input_cost']:.6f}")
        print(f"Output cost: ${result['aggregator_cost']['output_cost']:.6f}")
        print(f"Total cost: ${result['aggregator_cost']['total_cost']:.6f}")

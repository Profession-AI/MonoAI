from models._base_model import BaseModel
from models.model import Model
from models._prompt_executor import PromptExecutorMixin
from typing import List, Dict, Union
from typing import override
import asyncio
from prompts.prompt_chain import PromptChain
from prompts.prompt import Prompt

class CollectiveModel(BaseModel, PromptExecutorMixin):
    def __init__(
        self,
        models: List[Dict[str, str]],
        aggregator: Dict[str, str],
        count_tokens: bool = False,
        count_cost: bool = False
    ):
        """
        Initialize CollectiveModel with a list of models and an aggregator model.
        
        Args:
            models: List of dictionaries containing provider and model information
            aggregator: Dictionary containing aggregator model provider and model information
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
        prompt_text = str(prompt) if isinstance(prompt, (Prompt, PromptChain)) else prompt
        model_responses = "\n\n".join([
            f"Model {i+1} ({response['provider']} - {response['model']}):\n{response['output']}"
            for i, response in enumerate(responses)
        ])
        
        return f"""Please analyze the following responses from different models and provide a comprehensive answer:

Original Question: {prompt_text}

Model Responses:
{model_responses}

Please provide a well-reasoned response that takes into account all the information above."""

    def _extract_model_stats(self, responses: List[Dict]) -> Dict:
        """
        Extract statistics from model responses.
        
        Args:
            responses: List of responses from individual models
            
        Returns:
            Dictionary containing aggregated statistics
        """
        stats = {
            'total_tokens': 0,
            'total_cost': 0,
            'model_stats': []
        }
        
        for response in responses:
            model_stat = {
                'provider': response['provider'],
                'model': response['model']
            }
            
            if 'tokens' in response:
                model_stat['tokens'] = response['tokens']
                stats['total_tokens'] += response['tokens']['total_tokens']
            
            if 'cost' in response:
                model_stat['cost'] = response['cost']
                stats['total_cost'] += response['cost']['total_cost']
            
            stats['model_stats'].append(model_stat)
        
        return stats

    def _combine_results(
        self,
        prompt: Union[str, Prompt, PromptChain],
        model_responses: List[Dict],
        aggregator_response: Dict
    ) -> Dict:
        """
        Combine results from individual models and aggregator.
        
        Args:
            prompt: The original prompt or prompt chain
            model_responses: List of responses from individual models
            aggregator_response: Response from the aggregator model
            
        Returns:
            Combined results dictionary
        """
        result = {
            'output': aggregator_response['output'],
            'provider': 'collective',
            'model': 'aggregated',
            'stats': self._extract_model_stats(model_responses)
        }
        
        if 'tokens' in aggregator_response:
            result['tokens'] = aggregator_response['tokens']
            result['stats']['total_tokens'] += aggregator_response['tokens']['total_tokens']
        
        if 'cost' in aggregator_response:
            result['cost'] = aggregator_response['cost']
            result['stats']['total_cost'] += aggregator_response['cost']['total_cost']
        
        return result

    @override
    async def ask_async(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Ask all models and aggregate their responses asynchronously.
        
        Args:
            prompt: The prompt or prompt chain to process
            
        Returns:
            Dictionary containing aggregated response and statistics
        """
        # Get responses from all models
        model_responses = await asyncio.gather(*[
            self._execute_async(prompt, model._agent)
            for model in self._models
        ])
        
        # Process responses
        processed_responses = [
            self._process_prompt(
                prompt,
                response,
                model.provider_name,
                model.model_name,
                self._count_tokens,
                self._count_cost
            )
            for response, model in zip(model_responses, self._models)
        ]
        
        # Get aggregator response
        aggregator_prompt = self._format_aggregator_prompt(prompt, processed_responses)
        aggregator_response = await self._execute_async(aggregator_prompt, self._aggregator._agent)
        
        # Process aggregator response
        processed_aggregator = self._process_prompt(
            aggregator_prompt,
            aggregator_response,
            self._aggregator.provider_name,
            self._aggregator.model_name,
            self._count_tokens,
            self._count_cost
        )
        
        return self._combine_results(prompt, processed_responses, processed_aggregator)

    @override
    def ask(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Ask all models and aggregate their responses synchronously.
        
        Args:
            prompt: The prompt or prompt chain to process
            
        Returns:
            Dictionary containing aggregated response and statistics
        """
        return asyncio.run(self.ask_async(prompt))


if __name__ == "__main__":
    # Initialize with counting preferences
    collective_model = CollectiveModel(
        models=[
            {'provider': 'openai', 'model': 'gpt-4o-mini'},
            {'provider': 'deepseek', 'model': 'deepseek-chat'},
        ],
        aggregator={'provider': 'openai', 'model': 'gpt-4o-mini'},
        count_tokens=True,
        count_cost=True
    )
    
    # Example with simple prompt
    result = collective_model.ask("What is the capital of France?")
    print("\nSimple prompt result:")
    print(result["output"])
    print("\nStatistics:")
    print(f"Total tokens: {result['stats']['total_tokens']:,}")
    print(f"Total cost: ${result['stats']['total_cost']:.6f}")
    print("\nModel Statistics:")
    for stat in result['stats']['model_stats']:
        print(f"\n{stat['provider']} - {stat['model']}:")
        if 'tokens' in stat:
            print(f"Tokens: {stat['tokens']['total_tokens']:,}")
        if 'cost' in stat:
            print(f"Cost: ${stat['cost']['total_cost']:.6f}")
    
    # Example with prompt chain
    chain = PromptChain([
        "What is the capital of France?",
        "Based on the previous answer, what is its population?"
    ])
    result = collective_model.ask(chain)
    print("\nPrompt chain result:")
    print(result["output"])
    print("\nStatistics:")
    print(f"Total tokens: {result['stats']['total_tokens']:,}")
    print(f"Total cost: ${result['stats']['total_cost']:.6f}")
    print("\nModel Statistics:")
    for stat in result['stats']['model_stats']:
        print(f"\n{stat['provider']} - {stat['model']}:")
        if 'tokens' in stat:
            print(f"Tokens: {stat['tokens']['total_tokens']:,}")
        if 'cost' in stat:
            print(f"Cost: ${stat['cost']['total_cost']:.6f}")

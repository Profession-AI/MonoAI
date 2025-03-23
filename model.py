from pydantic_ai import Agent
from keys_manager import load_key
from typing import Sequence, Dict, Optional
from token_counter import TokenCounter
from token_cost import TokenCost


class Model:
    def __init__(
        self, 
        provider_name: str, 
        model_name: str, 
        system_prompt: str | Sequence[str] = (),
        count_tokens: bool = False,
        count_cost: bool = False
    ):
        """
        Initialize Model with provider, model name, and counting preferences.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            model_name: Name of the model (e.g., 'gpt-4', 'claude-3')
            system_prompt: System prompt or sequence of prompts
            count_tokens: Whether to count tokens for each request
            count_cost: Whether to calculate costs for each request
        """
        load_key(provider_name)
        self.provider_name = provider_name
        self.model_name = model_name
        self._count_tokens = count_tokens
        self._count_cost = count_cost
        self._agent = Agent(provider_name + ":" + model_name, system_prompt=system_prompt)

    def _post_process_response(self, question: str, answer: str) -> Dict:
        """
        Process the response and add optional token and cost information.
        
        Args:
            question: The input question
            answer: The model's answer
            
        Returns:
            Dictionary containing the response and optional stats
        """
        response = {
            "input": question, 
            "output": answer, 
            "model": {
                "provider": self.provider_name, 
                "name": self.model_name
            }
        }
        
        if self._count_tokens or self._count_cost:
            tokens = None
            if self._count_tokens:
                tokens = TokenCounter().count(self.model_name, question, answer)
                response["tokens"] = tokens
                
            if self._count_cost and tokens:
                cost = TokenCost().compute(
                    self.provider_name, 
                    self.model_name, 
                    tokens["input_tokens"], 
                    tokens["output_tokens"]
                )
                response["cost"] = cost
                
        return response
    
    async def ask_async(self, question: str) -> Dict:
        """
        Ask the model asynchronously.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the response and optional stats
        """
        answer = await self._agent.run(question)
        return self._post_process_response(question, answer.data)

    def ask(self, question: str) -> Dict:
        """
        Ask the model synchronously.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the response and optional stats
        """
        answer = self._agent.run_sync(question).data
        return self._post_process_response(question, answer)


if __name__ == "__main__":
    # Initialize with counting preferences
    model = Model(
        provider_name="openai",
        model_name="gpt-4o-mini",
        count_tokens=True,
        count_cost=True
    )
    
    # Get response with stats
    result = model.ask("What is the capital of France?")
    
    # Print results
    print("\nResponse:")
    print(result["output"])
    
    if "tokens" in result:
        print("\nToken Count:")
        print(f"Input tokens: {result['tokens']['input_tokens']:,}")
        print(f"Output tokens: {result['tokens']['output_tokens']:,}")
        print(f"Total tokens: {result['tokens']['total_tokens']:,}")
    
    if "cost" in result:
        print("\nCost:")
        print(f"Input cost: ${result['cost']['input_cost']:.6f}")
        print(f"Output cost: ${result['cost']['output_cost']:.6f}")
        print(f"Total cost: ${result['cost']['total_cost']:.6f}")

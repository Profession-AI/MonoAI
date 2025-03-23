import tiktoken
from typing import Dict
import json
from pathlib import Path

class TokenCost:

    _TOKEN_COST_FILE = "token_pricing.json"

    def compute(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """
        Compute token costs for input and output tokens.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            model: The model name (e.g., 'gpt-4', 'claude-3')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary containing input_cost, output_cost, and total_cost
            
        Raises:
            FileNotFoundError: If token_pricing.json is not found
            KeyError: If provider or model is not found in pricing file
            ValueError: If input_tokens or output_tokens are negative
            json.JSONDecodeError: If token_pricing.json is invalid JSON
        """
        # Validate input parameters
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        # Check if pricing file exists
        if not Path(self._TOKEN_COST_FILE).exists():
            raise FileNotFoundError(f"Pricing file '{self._TOKEN_COST_FILE}' not found")

        try:
            with open(self._TOKEN_COST_FILE, "r") as f:
                token_costs = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {self._TOKEN_COST_FILE}", e.doc, e.pos)

        # Validate provider exists
        if provider not in token_costs:
            raise KeyError(f"Provider '{provider}' not found in pricing file")

        # Validate model exists
        if model not in token_costs[provider]:
            raise KeyError(f"Model '{model}' not found for provider '{provider}'")

        # Validate pricing structure
        if not all(key in token_costs[provider][model] for key in ["input", "output"]):
            raise KeyError(f"Missing pricing information for model '{model}'")

        try:
            input_cost = round(input_tokens * token_costs[provider][model]["input"] / 1000, 8)
            output_cost = round(output_tokens * token_costs[provider][model]["output"] / 1000, 8)
            total_cost = round(input_cost + output_cost, 8)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error calculating costs: {str(e)}")

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
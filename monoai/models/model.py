from ._base_model import BaseModel
from ..keys.keys_manager import load_key
from ._response_processor import ResponseProcessorMixin
from ._prompt_executor import PromptExecutorMixin
from typing import Sequence, Dict, Union, AsyncGenerator
from ..tokens.token_counter import TokenCounter
from ..tokens.token_cost import TokenCost
from ..prompts.prompt_chain import PromptChain
from ..prompts.prompt import Prompt
from monoai.conf import Conf

class Model(BaseModel, ResponseProcessorMixin, PromptExecutorMixin):
    """
    Model class for interacting with AI language models.

    This module provides the Model class which serves as the primary interface for interacting
    with various AI language models (like GPT-4, Claude-3, etc.).

    Examples
    --------
    Basic usage:
    ```
    model = Model(provider="openai", model="gpt-4")
    response = model.ask("What is the capital of France?")
    ```

    With prompt:
    ```
    model = Model(
        provider="anthropic",
        model="claude-3",
    )
    prompt = Prompt(
        prompt="What is the capital of {country}?",
        prompt_data={"country": "France"},
        response_type=str
    )
    response = model.ask(prompt)
    ```
    """

    def __init__(
        self, 
        provider: str | None = None, 
        model: str | None = None, 
        system_prompt: str | Sequence[str] = (),
        count_tokens: bool = False, 
        count_cost: bool = False,
        max_tokens: int = None
    ):
        """
        Initialize a new Model instance.

        Parameters
        ----------
        provider : str
            Name of the provider (e.g., 'openai', 'anthropic')
        model : str
            Name of the model (e.g., 'gpt-4', 'claude-3')
        system_prompt : str | Sequence[str], optional
            System prompt or sequence of prompts
        count_tokens : bool, optional
            Whether to count tokens for each request
        count_cost : bool, optional
            Whether to calculate costs for each request
        max_tokens : int, optional
            Maximum number of tokens for each request
        """
        super().__init__(count_tokens, count_cost, max_tokens)
        
        if provider is None:
            provider = Conf()["base_model"]["provider"]
        if model is None:
            model = Conf()["base_model"]["model"]

        load_key(provider)

        self.provider = provider
        self.model = model
        self._web_search = False

    async def _ask_async(self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = {}) -> Dict:
        """
        Ask the model asynchronously.

        Parameters
        ----------
        prompt : Union[str, Prompt]
            The prompt to process
        metadata : Dict, optional
            Metadata to pass to the completion call

        Returns
        -------
        Dict
            Dictionary containing:
            - response: The model's response
            - prompt: The original prompt
            - model: Dictionary with provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)

        """
        response = await self._execute_async(prompt, metadata)
        return self._process_response(
            prompt,
            response,
        )

    
    async def ask_stream(self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = {}) -> AsyncGenerator[Dict, None]:
        """
        Ask the model with streaming response.

        Parameters
        ----------
        prompt : Union[str, Prompt, PromptChain]
            The prompt to process
        metadata : Dict, optional
            Metadata to pass to the completion call

        Yields
        ------
        Dict
            Streaming response chunks
        """
        yield {"provider":self.provider, "model":self.model}
        async for chunk in self._execute_stream(prompt, metadata):
            processed_chunk = self._process_chunk(chunk)
            if processed_chunk["delta"] is not None:
                yield processed_chunk



    def ask(self, prompt: Union[str, Prompt, PromptChain], metadata: Dict = {}) -> Dict:
        """
        Ask the model.

        Parameters
        ----------
        prompt : Union[str, Prompt]
            The prompt to process
        metadata : Dict, optional
            Metadata to pass to the completion call

        Returns
        -------
        Dict
            Dictionary containing:
            - response: The model's response
            - prompt: The original prompt
            - model: Dictionary with provider and model name
            - tokens: Token counts (if enabled)
            - cost: Cost calculation (if enabled)

        """
        response = self._execute(prompt, metadata)
        return self._process_response(
            prompt,
            response
        )
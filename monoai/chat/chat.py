from litellm import completion, acompletion
from monoai.keys.keys_manager import load_key
from monoai.chat.history import JSONHistory, SQLiteHistory, BaseHistory, HistorySummarizer
from monoai.models import Model

class Chat():
    """
    Chat class is responsible for handling the chat interface and messages history.
    
    Examples
    --------
    Basic usage:
    ```
    chat = Chat(provider="openai", model="gpt-4o-mini")
    response = chat.ask("2+2") # 4
    response = chat.ask("+2") # 6
    ```    

    With history:
    ```

    # Create a new chat
    chat = Chat(provider="openai", model="gpt-4o-mini", history_type="json")
    print(chat.chat_id) # 8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3
    response = chat.ask("Hello! I'm Giuseppe") # Hello!

    # Load a chat
    chat = Chat(provider="openai", model="gpt-4o-mini", history_type="json", chat_id="8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3")
    response = chat.ask("What's my name?") # Your name is Giuseppe
    ```

    With history summarizer:

    ```
    chat = Chat(provider="openai", 
                model="gpt-4o-mini", 
                history_type="json", 
                history_summarizer_provider="openai", 
                history_summarizer_model="gpt-4o-mini", 
                history_summarizer_max_tokens=100)
                
    response = chat.ask("Hello! I'm Giuseppe") # Hello!
    response = chat.ask("What's my name?") # Your name is Giuseppe
    ```
    """

    _HISTORY_MAP = {
        "json": JSONHistory,
        "sqlite": SQLiteHistory
    }

    def __init__(self, 
                 provider: str, 
                 model: str, 
                 system_prompt: str = "You are a helpful assistant.",
                 max_tokens: int = None,
                 history_type: str | BaseHistory = "json", 
                 history_summarizer_provider: str = None, 
                 history_summarizer_model: str = None,
                 history_summarizer_max_tokens: int = None,
                 chat_id: str = None):

        """
        Initialize a new Chat instance.

        Parameters
        ----------
        provider : str
            Name of the provider (e.g., 'openai', 'anthropic')
        model : str
            Name of the model (e.g., 'gpt-4', 'claude-3')
        system_prompt : str | Sequence[str], optional
            System prompt or sequence of prompts
        max_tokens : int, optional
            Maximum number of tokens for each request
        history_type : str | BaseHistory, optional
            The type of history to use for the chat.
        history_summarizer_provider : str, optional
            The provider of the history summarizer.
        history_summarizer_model : str, optional
            The model of the history summarizer.
        history_summarizer_max_tokens : int, optional
            The maximum number of tokens for the history summarizer.
        chat_id : str, optional
            The id of the chat to load, if not provided a new chat will be created
        """

        self._model = model
        self._max_tokens = max_tokens
        load_key(provider)
        self._history_summarizer = None

        if isinstance(history_type, str):
            self._history = self._HISTORY_MAP[history_type]()
        else:
            self._history = history_type

        if history_summarizer_provider is not None and history_summarizer_model is not None:
            self._history_summarizer = HistorySummarizer(Model(provider=history_summarizer_provider, 
                                                               model=history_summarizer_model,
                                                               max_tokens=history_summarizer_max_tokens))

        if chat_id is None:
            self.chat_id = self._history.new(system_prompt)
        else:
            self.chat_id = chat_id

    def ask(self, prompt: str, return_history: bool = False) -> str:
        
        """
        Ask the model a question.

        Parameters
        ----------
        prompt : str
            The question to ask the model
        return_history : bool, optional
            Whether to return the full history of the chat or only the response

        Returns
        -------
        str
            The response from the model
        list
            The full history of the chat if return_history is True
        """

        messages = self._history.load(self.chat_id)

        if self._history_summarizer is not None:
            summarized = self._history_summarizer.summarize(messages)
            messages.append({"role": "user", "content": prompt})
            ask_messages = [messages[0], messages[-1]]
            print(summarized)
            print(ask_messages[0])
            ask_messages[0]["content"] += "\n\nHere is the summary of the conversation so far:\n\n" + summarized
        else:
            messages.append({"role": "user", "content": prompt})
            ask_messages = messages

        response = completion(
            model=self._model,
            messages=ask_messages,
            max_tokens=self._max_tokens
        )
        
        response = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": response})
        
        self._history.store(self.chat_id, messages)
        if return_history:
            return messages
        else:
            return response
    
    async def ask_async(self, prompt: str):

        """
        Ask the model a question and stream the response.

        Parameters
        ----------
        prompt : str
            The question to ask the model

        Returns
        -------
        AsyncGenerator[str, None]
            A generator that yields the response from the model
        """
        messages = self._history.load(self.chat_id)
        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=self._max_tokens
        )

        response_text = ""
        async for part in response:
            part = part["choices"][0]["delta"]["content"] or ""
            response_text += part
            yield part
        messages.append({"role": "assistant", "content": response_text})
        self._history.store(self.chat_id, messages)




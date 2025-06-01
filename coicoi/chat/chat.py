from litellm import completion, acompletion
from coicoi.keys.keys_manager import load_key
from coicoi.chat.history import *

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
    chat = Chat(provider="openai", model="gpt-4o-mini", history_type="local")
    print(chat.chat_id) # 8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3
    response = chat.ask("Hello! I'm Giuseppe") # Hello!

    # Load a chat
    chat = Chat(provider="openai", model="gpt-4o-mini", history_type="local", chat_id="8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3")
    response = chat.ask("What's my name?") # Your name is Giuseppe
    ```

    With custom memory:

    ```
    chat = Chat(provider="openai", model="gpt-4o-mini", history_type="local", memory_type="custom")
    response = chat.ask("Hello! I'm Giuseppe") # Hello!
    response = chat.ask("What's my name?") # Your name is Giuseppe
    ```
    """

    _HISTORY_MAP = {
        "json": JSONHistory,
        "sqlite": SQLiteHistory
    }

    def __init__(self, provider: str, model: str, system_prompt: str = None, history_type: str | BaseHistory = "json", chat_id: str = None):

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
        history_type : str | BaseHistory, optional
            The type of history to use for the chat.
        chat_id : str, optional
            The id of the chat to load, if not provided a new chat will be created
        """

        if chat_id is not None and system_prompt is not None:
            raise ValueError("Cannot specify both chat_id and system_prompt. Use chat_id to load an existing chat or system_prompt to create a new one.")
        self.model = model
        load_key(provider)

        if isinstance(history_type, str):
            self._history = self._HISTORY_MAP[history_type]()
        else:
            self._history = history_type

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
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages
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
            model=self.model,
            messages=messages,
            stream=True
        )

        response_text = ""
        async for part in response:
            part = part["choices"][0]["delta"]["content"] or ""
            response_text += part
            yield part
        print(response_text)
        messages.append({"role": "assistant", "content": response_text})
        self._history.store(self.chat_id, messages)




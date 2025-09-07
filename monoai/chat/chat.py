from litellm import completion, acompletion
from monoai.keys.keys_manager import load_key
from monoai.chat.history import *
from monoai.models import Model
from monoai.conf.conf import Conf
from monoai.prompts.prompt import Prompt
import os
import base64
import json
import logging
from typing import Union, Optional, AsyncGenerator, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatError(Exception):
    """Custom exception for Chat-related errors."""
    pass

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

    # Create a new chat with JSON history
    chat = Chat(provider="openai", model="gpt-4o-mini", history="json")
    print(chat.chat_id) # 8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3
    response = chat.ask("Hello! I'm Giuseppe") # Hello!

    # Load a chat with JSON history
    chat = Chat(provider="openai", model="gpt-4o-mini", history="json", chat_id="8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3")
    response = chat.ask("What's my name?") # Your name is Giuseppe

    # Create a new chat with in-memory dictionary history
    chat = Chat(provider="openai", model="gpt-4o-mini", history="dict")
    print(chat.chat_id) # 8cc2bfa3-e9a0-4b82-b46e-3376cd220dd3
    response = chat.ask("Hello! I'm Giuseppe") # Hello!
    ```

    With history summarizer:

    ```
    chat = Chat(provider="openai", 
                model="gpt-4o-mini", 
                history="json", 
                history_summarizer_provider="openai", 
                history_summarizer_model="gpt-4o-mini", 
                history_summarizer_max_tokens=100)
                
    response = chat.ask("Hello! I'm Giuseppe") # Hello!
    response = chat.ask("What's my name?") # Your name is Giuseppe
    ```

    With metadata for observability:

    ```
    chat = Chat(provider="openai", model="gpt-4o-mini")
    
    # Pass metadata for tracking
    response = chat.ask(
        "Hello! I'm Giuseppe", 
        metadata={
            "user_id": "12345",
            "session_id": "abc-def-ghi",
            "feature": "chat"
        }
    )
    
    # Streaming with metadata
    async for chunk in chat.ask_stream(
        "Tell me a story",
        metadata={"user_id": "12345", "request_type": "story_generation"}
    ):
        print(chunk)
    ```
    """

    _HISTORY_MAP = {
        "json": JSONHistory,
        "sqlite": SQLiteHistory,
        "mongodb": MongoDBHistory,
        "dict": DictHistory
    }

    def __init__(self, 
                 model,
                 system_prompt: Optional[Union[Prompt, str]] = "You are an helpful assistant",
                 max_tokens: Optional[int] = None,
                 history: Union[str, BaseHistory] = "dict", 
                 history_last_n: Optional[int] = None,
                 history_path: Optional[str] = "histories",
                 history_summarizer_provider: Optional[str] = None, 
                 history_summarizer_model: Optional[str] = None,
                 history_summarizer_max_tokens: Optional[int] = None,
                 chat_id: Optional[str] = None) -> None:

        """
        Initialize a new Chat instance.

        Parameters
        ----------
        model : Model
            Model instance to use for the chat
        system_prompt : str | Prompt, optional
            System prompt or Prompt object
        max_tokens : int, optional
            Maximum number of tokens for each request
        history : str | BaseHistory, optional
            The type of history to use for the chat. Options: "json", "sqlite", "mongodb", "dict"
        history_last_n : int, optional
            The last n messages to keep in the history.
        history_path : str, optional
            The path to the history (not used for "dict" history type)
        history_summarizer_provider : str, optional
            The provider of the history summarizer.
        history_summarizer_model : str, optional
            The model of the history summarizer.
        history_summarizer_max_tokens : int, optional
            The maximum number of tokens for the history summarizer.
        chat_id : str, optional
            The id of the chat to load, if not provided a new chat will be created

        Raises
        ------
        ChatError
            If invalid parameters are provided or initialization fails
        """
        try:
            
            self._max_tokens = max_tokens
            self._model = model
            
            self._history_summarizer = None

            # Initialize history
            self._initialize_history(history, history_last_n, history_path)
            
            # Initialize history summarizer
            self._initialize_history_summarizer(
                history_summarizer_provider, 
                history_summarizer_model, 
                history_summarizer_max_tokens
            )

            # Process system prompt
            processed_system_prompt = self._process_system_prompt(system_prompt)

            # Initialize chat
            if chat_id is None:
                self.chat_id = self._history.new(processed_system_prompt)
            else:
                # Check if chat exists, if not create it
                self.chat_id = self._initialize_or_create_chat(chat_id, processed_system_prompt)
            
            self._metadata = {"session_id": self.chat_id}

        except Exception as e:
            logger.error(f"Failed to initialize Chat: {e}")
            raise ChatError(f"Chat initialization failed: {e}")

    def _initialize_or_create_chat(self, chat_id: str, system_prompt: str) -> str:
        """Initialize existing chat or create new one with specified ID.
        
        Parameters
        ----------
        chat_id : str
            The chat ID to check/create
        system_prompt : str
            System prompt to use if creating new chat
            
        Returns
        -------
        str
            The chat ID (either existing or newly created)
        """
        try:
            # Try to load existing chat
            existing_messages = self._history.load(chat_id)
            
            # Check if chat exists and has messages
            if existing_messages and len(existing_messages) > 0:
                logger.info(f"Loaded existing chat: {chat_id}")
                return chat_id
            else:
                # Chat doesn't exist, create it with the specified ID
                logger.info(f"Creating new chat with ID: {chat_id}")
                return self._create_chat_with_id(chat_id, system_prompt)
                
        except Exception as e:
            # If loading fails, assume chat doesn't exist and create it
            logger.info(f"Failed to load chat {chat_id}, creating new one: {e}")
            return self._create_chat_with_id(chat_id, system_prompt)
    
    def _create_chat_with_id(self, chat_id: str, system_prompt: str) -> str:
        """Create a new chat with a specific ID.
        
        Parameters
        ----------
        chat_id : str
            The specific chat ID to use
        system_prompt : str
            System prompt for the new chat
            
        Returns
        -------
        str
            The chat ID
        """
        try:
            # Create system message
            system_message = {"role": "system", "content": system_prompt}
            
            # Store the system message with the specific chat_id
            self._history.store(chat_id, [system_message])
            
            return chat_id
            
        except Exception as e:
            logger.error(f"Failed to create chat with ID {chat_id}: {e}")
            raise ChatError(f"Failed to create chat with ID {chat_id}: {e}")


    def _initialize_history(self, history: Union[str, BaseHistory], history_last_n: Optional[int], history_path: Optional[str]) -> None:
        """Initialize the history system."""
        try:
            if isinstance(history, str):
                if history == "dict":
                    # DictHistory doesn't need db_path
                    self._history = self._HISTORY_MAP[history](last_n=history_last_n)
                else:
                    self._history = self._HISTORY_MAP[history](last_n=history_last_n, path=history_path)
            else:
                self._history = history
        except Exception as e:
            raise ChatError(f"Failed to initialize history: {e}")

    def _initialize_history_summarizer(self, provider: Optional[str], model: Optional[str], max_tokens: Optional[int]) -> None:
        """Initialize the history summarizer."""
        if provider is not None and model is not None:
            try:
                self._history_summarizer = HistorySummarizer(
                    Model(provider=provider, model=model, max_tokens=max_tokens)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize history summarizer: {e}")

    def _process_system_prompt(self, system_prompt: Optional[Union[Prompt, str]]) -> str:
        """Process and load the system prompt."""
        try:
            if system_prompt is None:
                return self._load_default_system_prompt()
            elif isinstance(system_prompt, str) and system_prompt.endswith(".prompt"):
                return self._load_prompt_file(system_prompt)
            elif isinstance(system_prompt, Prompt):
                return str(system_prompt)
            else:
                return system_prompt
        except Exception as e:
            logger.warning(f"Failed to process system prompt: {e}")
            return ""

    def _load_default_system_prompt(self) -> str:
        """Load the default system prompt from file."""
        try:
            prompt_path = Conf()["prompts_path"]
            system_prompt_path = Path(prompt_path) / "system.prompt"
            
            if system_prompt_path.exists():
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                logger.info("Default system prompt file not found, using empty prompt")
                return ""
        except Exception as e:
            logger.warning(f"Failed to load default system prompt: {e}")
            return ""

    def _load_prompt_file(self, prompt_file: str) -> str:
        """Load a prompt from a file."""
        try:
            prompt_path = Conf()["prompts_path"]
            full_path = Path(prompt_path) / prompt_file
            
            if not full_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {full_path}")
                
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ChatError(f"Failed to load prompt file {prompt_file}: {e}")

    def _process_file_attachment(self, prompt: str, file: Optional[Union[str, bytes]], file_type: Optional[str]) -> Union[str, List[Dict[str, Any]]]:
        """Process file attachment and return modified prompt."""
        if file is None:
            return prompt
            
        try:
            # Determine file type
            if file_type is None and isinstance(file, str):
                file_type = file.split(".")[-1].lower()
                
            if not file_type:
                raise ChatError("File type could not be determined")
                
            conf = Conf()
            
            # Handle text files
            if file_type in conf["supported_files"]["text"]:
                return self._process_text_file(prompt, file)
            # Handle image files
            elif file_type in conf["supported_files"]["image"]:
                return self._process_image_file(prompt, file)
            else:
                raise ChatError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise ChatError(f"Failed to process file attachment: {e}")

    def _process_text_file(self, prompt: str, file: Union[str, bytes]) -> str:
        """Process text file attachment."""
        try:
            if isinstance(file, str):
                # Handle as file path or URL
                if os.path.exists(file):
                    with open(file, "r", encoding="utf-8") as f:
                        file_content = f.read()
                else:
                    # Assume it's a URL or remote file
                    file_content = file
            else:
                # Handle as bytes
                file_content = file.decode("utf-8")
                
            return prompt + Conf()["default_prompt"]["file"] + file_content
            
        except Exception as e:
            raise ChatError(f"Failed to process text file: {e}")

    def _process_image_file(self, prompt: str, file: Union[str, bytes]) -> List[Dict[str, Any]]:
        """Process image file attachment."""
        try:
            if isinstance(file, str):
                # Handle as file path or URL
                if os.path.exists(file):
                    # Convert local file to base64
                    with open(file, "rb") as f:
                        file_bytes = f.read()
                    image_url = f"data:image/jpeg;base64,{base64.b64encode(file_bytes).decode('utf-8')}"
                else:
                    # Assume it's already a URL
                    image_url = file
            else:
                # Handle as bytes
                image_url = f"data:image/jpeg;base64,{base64.b64encode(file).decode('utf-8')}"
                
            return [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]
            
        except Exception as e:
            raise ChatError(f"Failed to process image file: {e}")

    def _prepare_messages(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prepare messages for the API call."""
        try:
            messages = self._history.load(self.chat_id)
            messages = [{'role': d.get('role'), 'content': d.get('content')} for d in messages]

            messages.append({"role": "user", "content": prompt})

            # Apply history summarization if enabled
            if self._history_summarizer is not None:
                return self._apply_history_summarization(messages)
            else:
                return messages
                
        except Exception as e:
            raise ChatError(f"Failed to prepare messages: {e}")

    def _apply_history_summarization(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply history summarization to messages."""
        try:
            summarized = self._history_summarizer.summarize(messages)
            ask_messages = [messages[0], messages[-1]]
            ask_messages[0]["content"] += Conf()["default_prompt"]["summary"] + summarized
            return ask_messages
        except Exception as e:
            logger.warning(f"History summarization failed, using full history: {e}")
            return messages

    def _store_messages(self, messages: List[Dict[str, Any]], response_content: str) -> None:
        """Store messages in history."""
        try:
            messages.append({"role": "assistant", "content": response_content})
            self._history.store(self.chat_id, messages[-2:])
        except Exception as e:
            logger.error(f"Failed to store messages: {e}")


    def ask(self, prompt: str, file: Optional[Union[str, bytes]] = None, file_type: Optional[str] = None, return_history: bool = False, metadata: Optional[Dict[str, Any]] = {}) -> Union[str, List[Dict[str, Any]]]:
        """
        Ask the model a question.

        Parameters
        ----------
        prompt : str
            The question to ask the model
        file : str | bytes, optional
            The file to attach to the message, if it's a string it will be treated as a local path or remote url to a file
        file_type : str, optional
            The type of the file
        return_history : bool, optional
            Whether to return the full history of the chat or only the response
        metadata : Dict[str, Any], optional
            Metadata to pass to the model for observability and tracking

        Returns
        -------
        str | List[Dict[str, Any]]
            The response from the model or the full history if return_history is True

        Raises
        ------
        ChatError
            If the request fails or parameters are invalid
        """

        metadata = self._metadata | metadata

        try:
            # Process file attachment
            processed_prompt = self._process_file_attachment(prompt, file, file_type)
            
            # Prepare messages
            messages = self._prepare_messages(processed_prompt)
            
            # Make API call
            response = self._model.ask(messages, metadata=metadata)
            response_content = response["response"]
            
            # Store messages
            self._store_messages(messages, response_content)
            
            response = {
                "chat_id": self.chat_id,
                "response": response_content,
                "model": {
                    "provider": self._model.provider,
                    "name": self._model.model
                }
            }

            if return_history:
                response["history"] = messages
            
            return response
                
        except Exception as e:
            logger.error(f"Ask request failed: {e}")
            raise ChatError(f"Ask request failed: {e}")

    async def ask_stream(self, prompt: str, file: Optional[Union[str, bytes]] = None, file_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Ask the model a question and stream the response.

        Parameters
        ----------
        prompt : str
            The question to ask the model
        file : str | bytes, optional
            The file to attach to the message
        file_type : str, optional
            The type of the file
        metadata : Dict[str, Any], optional
            Metadata to pass to the model for observability and tracking

        Yields
        ------
        str
            JSON string containing streaming response chunks

        Raises
        ------
        ChatError
            If the request fails or parameters are invalid
        """

        metadata = self._metadata | metadata

        try:
            # Process file attachment
            processed_prompt = self._process_file_attachment(prompt, file, file_type)
            
            # Prepare messages
            messages = self._prepare_messages(processed_prompt)
            
            # Make streaming API call
            response = self._model.ask_stream(messages, metadata=metadata)
            yield {"chat_id": self.chat_id}
            response_text = ""
            async for chunk in response: 
                if "delta" in chunk:
                    response_text += chunk["delta"]
                yield chunk

            # Store messages
            self._store_messages(messages, response_text)
            
        except Exception as e:
            logger.error(f"Stream request failed: {e}")
            raise ChatError(f"Stream request failed: {e}")

    async def ask_async(self, prompt: str, file: Optional[Union[str, bytes]] = None, file_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Ask the model a question and stream the response asynchronously.

        Parameters
        ----------
        prompt : str
            The question to ask the model
        file : str | bytes, optional
            The file to attach to the message
        file_type : str, optional
            The type of the file
        metadata : Dict[str, Any], optional
            Metadata to pass to the model for observability and tracking

        Yields
        ------
        str
            Response chunks as strings

        Raises
        ------
        ChatError
            If the request fails or parameters are invalid
        """

        metadata = self._metadata | metadata

        try:
            # Process file attachment
            processed_prompt = self._process_file_attachment(prompt, file, file_type)
            
            # Prepare messages
            messages = self._prepare_messages(processed_prompt)
            
            # Make streaming API call
            response = await acompletion(
                model=self._model,
                messages=messages,
                stream=True,
                max_tokens=self._max_tokens,
                metadata=metadata
            )

            response_text = ""
            async for part in response:
                content = part["choices"][0]["delta"]["content"] or ""
                response_text += content
                yield content
                
            # Store messages
            self._store_messages(messages, response_text)
            
        except Exception as e:
            logger.error(f"Async request failed: {e}")
            raise ChatError(f"Async request failed: {e}")

    def get_chat_info(self) -> Dict[str, Any]:
        """
        Get information about the current chat.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing chat information
        """
        return {
            "chat_id": self.chat_id,
            "provider": self._provider,
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "has_history_summarizer": self._history_summarizer is not None
        }

    def clear_history(self) -> None:
        """
        Clear the chat history.

        Raises
        ------
        ChatError
            If clearing history fails
        """
        try:
            self._history.clear(self.chat_id)
            logger.info(f"Cleared history for chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise ChatError(f"Failed to clear history: {e}")
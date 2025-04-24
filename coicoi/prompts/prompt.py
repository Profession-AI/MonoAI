from ._prompt_parser import PromptParser



class Prompt:
    """
    A class to handle text prompts with optional response type specification.
    
    The Prompt class can be initialized either with a direct prompt string or by loading
    a prompt from a .prompt file. It supports response type specification and
    data formatting through Python's string formatting.

    .prompt format
    --------
    Simple prompt
    ```
    <prompt>
        Hello, coicoi!
    </prompt>
    ```
    In this case you could also use plain text without the prompt tag:
    ```
    Hello, coicoi!
    ```
    Prompt with data formatting
    ```
    <prompt>
        Hello, {name}!
    </prompt>
    ```
    Prompt with response type
    ```
    <prompt response_type="int">
        What is 2+2?
    </prompt>
    ```

    Examples
    --------
    Simple prompt with response type:
    ```
    prompt = Prompt(prompt="What is 2+2?", response_type=int)
    ```
    Prompt with data formatting:
    ```
    prompt = Prompt(
        prompt="What is the capital of {country}?",
        prompt_data={"country": "France"},
        response_type=str
    )
    ```
    Load prompt from file:
    ```
    prompt = Prompt(prompt_id="math_question")
    ```
    math_question.prompt
    ```
    <prompt response_type="int">
        What is 2+2?
    </prompt>
    ```
    """

    def __init__(self, 
                 prompt_id: str | None = None,
                 prompt_data: dict | None = None,
                 prompt: str | None = None, 
                 response_type: type | None = None):
        """
        Initialize a new Prompt instance.

        Parameters
        ----------
        prompt_id : str, optional
            A .prompt file name for loading a prompt from file
        prompt_data : dict, optional
            Dictionary of values for formatting the prompt
        prompt : str, optional
            Direct prompt text if prompt_id is not provided
        response_type : type, optional
            Expected type of the response

        Raises
        ------
        ValueError
            If neither prompt_id nor prompt is provided
        """
        prompt_response_type = None
        if prompt_id is not None:
            self._prompt, prompt_response_type = PromptParser().parse(prompt_id)
        elif prompt is not None:
            self._prompt = prompt
        else:
            raise ValueError("Either prompt_id or prompt must be provided")

        if prompt_data is not None:
            self._prompt = self._prompt.format(**prompt_data)

        if prompt_response_type is not None:
            self.response_type = prompt_response_type
        else:
            self.response_type = response_type

    def __str__(self) -> str:
        """
        Get the string representation of the prompt.

        Returns
        -------
        str
            The prompt text
        """
        return self._prompt
    
    def __repr__(self) -> str:
        """
        Get the official string representation of the prompt.

        Returns
        -------
        str
            The prompt text
        """
        return self.__str__()

from ._prompt_parser import PromptParser

class Prompt:
    def __init__(self, 
                 prompt_id: str | None = None,
                 prompt_data: dict | None = None,
                 prompt: str | None = None, 
                 output_type: type | None = None):
        
        if prompt_id is not None:
            self._prompt = PromptParser().parse(prompt_id)
        elif prompt is not None:
            self._prompt = prompt
        else:
            raise ValueError("Either prompt_id or prompt must be provided")

        if prompt_data is not None:
            self._prompt = self._prompt.format(**prompt_data)

        self.output_type = output_type

    def __str__(self):
        return self._prompt
    
    def __repr__(self):
        return self.__str__()

from pydantic_ai.agent import RunResultDataT

class Prompt:
    def __init__(self, prompt: str, result_type: type[RunResultDataT] | None = None):
        self._prompt = prompt
        self.result_type = result_type

    def __str__(self):
        return self._prompt
    
    def __repr__(self):
        return self.__str__()

class Prompt:
    def __init__(self, prompt: str):
        self._prompt = prompt

    def __str__(self):
        return self._prompt
    
    def __repr__(self):
        return self.__str__()

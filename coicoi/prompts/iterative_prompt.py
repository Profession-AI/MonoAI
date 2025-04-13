from typing import List
from .prompt import Prompt

class IterativePrompt(Prompt):

    def __init__(self, prompt:str, iter_context:List[str], prompt_memory:str=""):
        self._prompt = prompt.replace("{{", "{").replace("}}", "}")
        self._iter_context = iter_context
        self.size = len(iter_context)
        self._prompt_memory = prompt_memory.replace("{{", "{").replace("}}", "}")
        self.has_memory = prompt_memory!=""

    def format(self, index: int, context:str="") -> str:

        prompt = self._prompt.format(context=self._iter_context[index])
        
        if self.has_memory and index > 0:
            prompt += "\n\n"+self._prompt_memory.format(context=context)

        return prompt

    def __str__(self):
        return self._prompt

    def __repr__(self):
        return self.__str__()

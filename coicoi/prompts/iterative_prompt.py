from typing import List
from .prompt import Prompt
from ._prompt_parser import PromptParser
class IterativePrompt(Prompt):

    def __init__(self, 
                 prompt_id:str=None,
                 prompt:str=None,
                 prompt_data:dict=None,
                 iter_data:List[str]=None, 
                 prompt_memory:str=""):
        if prompt_id is not None:
            self._prompt, prompt_memory = PromptParser().parse(prompt_id)
        elif prompt is not None:
            self._prompt = prompt
        else:       
            raise ValueError("Either prompt_id or prompt must be provided")
        if prompt_data is not None:
            self._prompt = self._prompt.format(**prompt_data)
        self._iter_data = iter_data
        self.size = len(iter_data)
        self._prompt_memory = prompt_memory.replace("{{", "{").replace("}}", "}")
        self.has_memory = prompt_memory!=""        

    def format(self, index: int, context:str="") -> str:

        prompt = self._prompt.format(data=self._iter_data[index])
        if self.has_memory and index > 0:
            prompt += "\n\n"+self._prompt_memory.format(data=context)
        return prompt

    def __str__(self):
        return self._prompt

    def __repr__(self):
        return self.__str__()

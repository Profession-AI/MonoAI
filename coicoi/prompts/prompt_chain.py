from typing import List
from .prompt import Prompt
from ._prompt_parser import PromptParser
class PromptChain(Prompt):

    def __init__(self, 
                 promptchain_id: str=None,
                 prompts_data: list[dict]=None,
                 prompts: List[Prompt]=None, 
                 ):
        if promptchain_id is not None:
            self._prompts = PromptParser().parse(promptchain_id)
            for i in range(len(self._prompts)):
                self._prompts[i] = Prompt(prompt=self._prompts[i], prompt_data=prompts_data[i])
        elif prompts is not None:
            self._prompts = prompts
        else:
            raise ValueError("Either promptchain_id or prompts must be provided")
        self.size = len(self._prompts)

    def format(self, index: int, context: str | None = None) -> str:
        if context is None:
            return str(self._prompts[index])
        else:
            return str(self._prompts[index]) + "\n\n" + context

    def __str__(self):
        return "\n".join([str(prompt) for prompt in self._prompts])

    def __repr__(self):
        return self.__str__()

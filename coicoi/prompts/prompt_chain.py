from typing import List
from .prompt import Prompt

class PromptChain(Prompt):

    def __init__(self, prompts: List[str]):
        self._prompts = prompts
        self.size = len(prompts)

    def format(self, index: int, context: str | None = None) -> str:
        if context is None:
            return self._prompts[index]
        else:
            return self._prompts[index] + "\n\n" + context

    def __str__(self):
        return "\n".join(self._prompts)

    def __repr__(self):
        return self.__str__()

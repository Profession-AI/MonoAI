from ..models import MultiModel
from typing import List, Dict

class MultiAgent(MultiModel):

    def __init__(self, models: List[Dict[str, str]], tools):
        super().__init__(models)
        for model in self._models:
            for tool in tools:
                model._agent._register_tool(tool)

    def run(self, prompt: str):
        return super().ask(prompt)

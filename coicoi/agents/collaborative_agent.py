from ..models import CollaborativeModel
from typing import List, Dict

class CollaborativeAgent(CollaborativeModel):

    def __init__(
        self,
        models: List[Dict[str, str]],
        aggregator: Dict[str, str],
        tools = [],
        count_tokens: bool = False,
        count_cost: bool = False
    ):
        super().__init__(models, aggregator, count_tokens, count_cost)

        for model in self._multi_model._models:
            for tool in tools:
                model._agent._register_tool(tool)

    def run(self, prompt: str):
        return super().ask(prompt)

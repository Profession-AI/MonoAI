from ..models import CollaborativeModel
from typing import List, Dict, Union, Callable
from ..tools._base_tools import BaseTool

class CollaborativeAgent(CollaborativeModel):

    def __init__(
        self,
        models: List[Dict[str, str]],
        aggregator: Dict[str, str],
        tools: List[Union[BaseTool, Callable]] = [],
        count_tokens: bool = False,
        count_cost: bool = False
    ):
        super().__init__(models, aggregator, count_tokens, count_cost)

        tools_functions = [tool.get_tool() if isinstance(tool, BaseTool) else tool for tool in tools]

        for model in self._multi_model._models:
            for tool in tools_functions:
                model._agent._register_tool(tool)

    def run(self, prompt: str):
        return super().ask(prompt)

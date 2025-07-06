from ..models import Model

class Agent(Model):

    def __init__(self, provider: str, model: str, tools):
        super().__init__(provider, model)
        for tool in tools:
            self._agent._register_tool(tool)

    def run(self, prompt: str):
        return super().ask(prompt)

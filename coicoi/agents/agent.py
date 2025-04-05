from ..models import Model

class Agent(Model):

    def __init__(self, provider_name: str, model_name: str, tools):
        super().__init__(provider_name, model_name)
        for tool in tools:
            self._agent._register_tool(tool)

    def run(self, prompt: str):
        return super().ask(prompt)

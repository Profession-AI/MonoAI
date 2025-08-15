from ..models import Model
from .agentic_loop import FunctionCallAgenticLoop

class Agent():

    def __init__(self, model:Model, tools=None, paradigm="function_call"):
        
        self._model = model
        self._model._add_tools(tools)
        self._available_tools = {}
        
        for tool in tools:
            self._available_tools[tool.__name__]=tool
        
        if paradigm=="function_call":
            self._loop = FunctionCallAgenticLoop(self._model, self._available_tools)
        elif paradigm=="react":
            self._loop = ReactAgenticLoop(self._model, self._available_tools)

            
    def run(self, prompt: str):
        return self._loop.start(prompt)





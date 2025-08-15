from ..models import Model
from .agentic_loop import FunctionCallingAgenticLoop, ReactAgenticLoop

class Agent():

    def __init__(self, model:Model, tools=None, paradigm="function_call", debug=False, max_iter=None):
        
        self._model = model
        self._model._add_tools(tools)
        self._available_tools = {}
        
        for tool in tools:
            self._available_tools[tool.__name__]=tool

        loop_kwargs = self._model, self._available_tools, debug, max_iter
        
        if paradigm=="function_call":
            self._loop = FunctionCallingAgenticLoop(*loop_kwargs)
        elif paradigm=="react":
            self._loop = ReactAgenticLoop(*loop_kwargs)

            
    def run(self, prompt: str):
        return self._loop.start(prompt)





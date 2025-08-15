from ..models import Model
from .agentic_loop import FunctionCallingAgenticLoop, ReactAgenticLoop, ReactWithFCAgenticLoop

class Agent():

    def __init__(self, model:Model, tools=None, paradigm="function_calling", debug=False, max_iter=None):
        
        self._model = model

        loop_kwargs = self._model, tools, debug, max_iter
        
        if paradigm=="function_calling":
            self._loop = FunctionCallingAgenticLoop(*loop_kwargs)
        elif paradigm=="react":
            self._loop = ReactAgenticLoop(*loop_kwargs)
        elif paredigm=="react_with_function_calling":
            self.loop = ReactWithFCAgenticLoop(*loop_kwargs)
            
    def run(self, prompt: str):
        return self._loop.start(prompt)





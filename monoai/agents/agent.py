from ..models import Model
from .agentic_loop import (
    FunctionCallingAgenticLoop, 
    ReactAgenticLoop, 
    ReactWithFCAgenticLoop,
    PlanAndExecuteAgenticLoop,
    ProgrammaticAgenticLoop,
    ReflexionAgenticLoop,
    SelfAskAgenticLoop,
    SelfAskWithSearchLoop
)

class Agent():

    def __init__(self, model:Model, tools=None, paradigm="function_calling", agent_prompt=None, debug=False, max_iter=None):
        
        self._model = model

        loop_kwargs = self._model, tools, agent_prompt, debug, max_iter
        
        if paradigm=="function_calling":
            self._loop = FunctionCallingAgenticLoop(*loop_kwargs)
        elif paradigm=="react":
            self._loop = ReactAgenticLoop(*loop_kwargs)
        elif paradigm=="react_with_function_calling":
            self._loop = ReactWithFCAgenticLoop(*loop_kwargs)
        elif paradigm=="plan-and-execute":
            self._loop = PlanAndExecuteAgenticLoop(*loop_kwargs)
        elif paradigm=="programmatic":
            self._loop = ProgrammaticAgenticLoop(*loop_kwargs)
        elif paradigm=="reflexion":
            self._loop = ReflexionAgenticLoop(*loop_kwargs)
        elif paradigm=="self_ask":
            self._loop = SelfAskAgenticLoop(*loop_kwargs)
        elif paradigm=="self_ask_with_search":
            self._loop = SelfAskWithSearchLoop(*loop_kwargs)

            
    def run(self, prompt: str):
        return self._loop.start(prompt)





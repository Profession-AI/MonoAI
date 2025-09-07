from ..models import Model
from .agentic_loop import (
    FunctionCallingAgenticLoop, 
    ReactAgenticLoop, 
    ReactWithFCAgenticLoop,
    PlanAndExecuteAgenticLoop,
    ProgrammaticAgenticLoop,
    ReflexionAgenticLoop,
    SelfAskAgenticLoop,
    SelfAskWithSearchLoop,
    _AgenticLoop
)

from ..prompts import Prompt


class Agent:
    """AI Agent that implements different reasoning paradigms.
    
    This class provides a unified interface for creating and using AI agents
    with different reasoning paradigms. The agent can be configured to use
    function calling, ReAct, plan-and-execute, and other agentic approaches,
    or a custom paradigm.
    
    The Agent class acts as a high-level wrapper around specific agentic loop
    implementations, providing a consistent API regardless of the underlying
    reasoning pattern. It handles paradigm selection, tool registration,
    and streaming configuration.
    
    Supported Paradigms:
    - Function Calling: Native OpenAI function calling
    - ReAct: Reasoning and Acting pattern
    - Plan-and-Execute: Planning then execution
    - Programmatic: Code generation and execution
    - Reflexion: Self-reflective reasoning
    - Self-Ask: Self-questioning approach
    - Self-Ask with Search: Self-ask with web search capabilities
    
    """
    
    def __init__(self, model: Model, tools=None, paradigm="function_calling", 
                 agent_prompt=None, debug=False, max_iter=None, native_web_search=None):
        """Initialize the agent with the specified model and configuration.
        
        This constructor sets up an AI agent with the chosen reasoning paradigm,
        registers any provided tools, and configures execution parameters.
        
        Parameters
        ----------
        model : Model
            AI model instance to use for agent execution
        tools : list, optional
            List of available tools for the agent. Each tool should be a
            callable function. Default is None.
        paradigm : str or _AgenticLoop, optional
            Reasoning paradigm to use. Can be:
            
            **Predefined strings:**
            - "function_calling": Uses OpenAI native function calling
            - "react": Standard ReAct approach
            - "react_with_function_calling": Combines ReAct and function calling
            - "plan-and-execute": Plan-and-execute paradigm
            - "programmatic": Programmatic approach
            - "reflexion": Reflexion paradigm with self-reflection
            - "self_ask": Self-ask paradigm
            - "self_ask_with_search": Self-ask with web search capabilities
            
            **Custom object:**
            - An instance of a class derived from _AgenticLoop
            
            Default is "function_calling".
        agent_prompt : str, optional
            Custom prompt for the agent. If None, uses the default prompt
            for the chosen paradigm. Default is None.
        debug : bool, optional
            Flag to enable debug output during execution.
            Default is False.
        max_iter : int, optional
            Maximum number of iterations allowed for the agent.
            If None, there are no limits. Default is None.
        native_web_search : str, optional
            Native web search capability level. Must be one of:
            "low", "medium", or "high". Default is None.
        
        Raises
        ------
        ValueError
            If the specified paradigm is not supported or invalid
        TypeError
            If a custom object is passed that doesn't derive from _AgenticLoop
        """
        self._model = model

        if native_web_search is not None and native_web_search not in ["low", "medium", "high"]:
            raise ValueError("native_web_search must be 'low', 'medium' or 'high'")
        
        self._model._web_search = native_web_search

        # Gestione paradigma personalizzato
        if isinstance(paradigm, _AgenticLoop):
            # Verifica che l'oggetto personalizzato sia valido
            if not hasattr(paradigm, 'start') or not callable(paradigm.start):
                raise TypeError("Il paradigma personalizzato deve avere un metodo 'start' callable")
            self._loop = paradigm
        else:
            # Paradigmi predefiniti
            loop_kwargs = self._model, agent_prompt, debug, max_iter
            
            if paradigm == "function_calling":
                self._loop = FunctionCallingAgenticLoop(*loop_kwargs)
            elif paradigm == "react":
                self._loop = ReactAgenticLoop(*loop_kwargs)
            elif paradigm == "react_with_function_calling":
                self._loop = ReactWithFCAgenticLoop(*loop_kwargs)
            elif paradigm == "plan-and-execute":
                self._loop = PlanAndExecuteAgenticLoop(*loop_kwargs)
            elif paradigm == "programmatic":
                self._loop = ProgrammaticAgenticLoop(*loop_kwargs)
            elif paradigm == "reflexion":
                self._loop = ReflexionAgenticLoop(*loop_kwargs)
            elif paradigm == "self_ask":
                self._loop = SelfAskAgenticLoop(*loop_kwargs)
            elif paradigm == "self_ask_with_search":
                self._loop = SelfAskWithSearchLoop(*loop_kwargs)
            else:
                raise ValueError(f"Paradigma '{paradigm}' non supportato. "
                               f"Paradigmi disponibili: function_calling, react, "
                               f"react_with_function_calling, plan-and-execute, "
                               f"programmatic, reflexion, self_ask, self_ask_with_search, "
                               f"oppure un oggetto personalizzato derivato da _AgenticLoop")
        
        if tools is not None:
            self._loop.register_tools(tools)
        
    def run(self, prompt: str | Prompt):
        """Execute the agent with the specified prompt.
        
        This method processes a user prompt through the agent's reasoning
        paradigm, returning a structured response that includes the reasoning
        process and final answer.
        
        Parameters
        ----------
        prompt : str or Prompt
            User prompt or query to process. Can be a string or a Prompt object.
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing:
            - prompt: Original user query
            - iterations: List of processed reasoning iterations
            - response: Final response (if available)
        
        Notes
        -----
        This method delegates execution to the specific agentic loop
        configured during initialization. The exact behavior depends on
        the chosen paradigm (predefined or custom).
        
        The response structure may vary slightly depending on the paradigm:
        - Function calling: Includes tool call details
        - ReAct: Includes thought-action-observation cycles
        - Plan-and-execute: Includes planning and execution phases
        - Other paradigms: May include paradigm-specific information
        """
        
        return self._loop.start(prompt)
    
    def enable_streaming(self, stream_callback=None):
        """Enable streaming responses for this agent.
        
        When streaming is enabled, the agent will call the provided callback
        function with each content chunk as it's generated, allowing for
        real-time processing and display of the AI's response.
        
        Parameters
        ----------
        stream_callback : callable, optional
            Callback function to handle streaming content chunks.
            If None, uses a default callback that prints content to console.
            The callback receives plain text content strings.
        
        Raises
        ------
        AttributeError
            If the current paradigm doesn't support streaming
        """
        if hasattr(self._loop, 'enable_streaming'):
            self._loop.enable_streaming(stream_callback)
        else:
            raise AttributeError(f"Paradigm '{self._loop.__class__.__name__}' doesn't support streaming")
    
    def disable_streaming(self):
        """Disable streaming responses for this agent.
        
        After calling this method, the agent will use standard (non-streaming)
        model execution for all subsequent requests.
        
        Raises
        ------
        AttributeError
            If the current paradigm doesn't support streaming
        """
        if hasattr(self._loop, 'disable_streaming'):
            self._loop.disable_streaming()
        else:
            raise AttributeError(f"Paradigm '{self._loop.__class__.__name__}' doesn't support streaming")
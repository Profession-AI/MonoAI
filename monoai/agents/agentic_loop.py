"""
Agentic Loop Implementation with Streaming Support

This module provides a comprehensive set of agentic loop implementations that support
real-time streaming responses during the entire generation phase. The agents are designed
to work with various AI paradigms and can process user queries through iterative reasoning,
tool usage, and structured output generation.

Key Features:
- Real-time streaming of AI responses
- Multiple agent paradigms (React, Function Calling, Plan-and-Execute, etc.)
- Step-based reasoning format for transparent decision making
- Tool integration and execution
- Configurable iteration limits and debugging
- Asynchronous streaming with proper resource management

Agent Paradigms:
The agents use a structured step-based format for reasoning:
- Thought N: <reasoning process and analysis>
- Action N: {"name": "tool_name", "arguments": {...}}
- Observation N: <tool execution result>
- Final answer: <conclusive response to user query>

Streaming Architecture:
Streaming allows real-time processing of AI responses as they are generated, enabling:
- Immediate feedback to users
- Progress monitoring during long operations
- Better user experience with responsive interfaces
- Debugging and monitoring of agent reasoning

Usage Examples:

1. Basic React Agent with Streaming:
    ```python
    from monoai.agents import Agent
    
    agent = Agent(model, paradigm="react")
    agent.enable_streaming()  # Uses default console output
    result = agent.run("What is the capital of France?")
    ```

2. Custom Streaming Handler:
    ```python
    from monoai.agents import Agent
    import json
    
    def custom_stream_handler(content):
        # Process streaming content in real-time
        print(f"Streaming: {content}", end='', flush=True)
    
    agent = Agent(model, paradigm="react")
    agent.enable_streaming(custom_stream_handler)
    result = agent.run("Your query here")
    ```

3. Function Calling Agent:
    ```python
    from monoai.agents import Agent
    from monoai.tools import search_web
    
    agent = Agent(model, paradigm="function_calling")
    agent.register_tools([search_web])
    agent.enable_streaming()
    result = agent.run("Search for recent AI news")
    ```

4. Plan and Execute Agent:
    ```python
    agent = Agent(model, paradigm="plan-and-execute")
    agent.enable_streaming()
    result = agent.run("Create a detailed project plan")
    ```

Available Agent Types:
- FunctionCallingAgenticLoop: Native OpenAI function calling
- ReactAgenticLoop: ReAct reasoning pattern
- ReactWithFCAgenticLoop: Hybrid ReAct + Function Calling
- ProgrammaticAgenticLoop: Code generation and execution
- PlanAndExecuteAgenticLoop: Planning then execution pattern
- ReflexionAgenticLoop: Self-reflection and improvement
- SelfAskAgenticLoop: Self-questioning approach
- SelfAskWithSearchLoop: Self-ask with web search capabilities

Streaming Callback Format:
The streaming callback receives content as plain text strings. For advanced use cases,
you can access the raw streaming data through the model's streaming methods.

Error Handling:
- Automatic fallback to non-streaming mode on errors
- Proper cleanup of async resources
- Configurable debug output
- Iteration limits to prevent infinite loops

Thread Safety:
This implementation is designed for single-threaded use. For concurrent access,
create separate agent instances for each thread or process.
"""

import json
import inspect
import re
from typing import Any, Dict, List, Optional, Callable
from ..prompts import Prompt


class _FunctionCallingMixin:
    """Mixin for handling OpenAI function calling tool execution.
    
    This mixin provides methods for managing tool calls in OpenAI's native
    function calling format, converting tool responses into standardized messages
    that can be used in the conversation flow.
    
    The mixin handles the execution of tools registered with the agent and
    formats their responses according to OpenAI's message format specification.
    """
    
    def _call_tool(self, tool_call: Any) -> Dict[str, str]:
        """Execute a tool call and return the formatted response.
        
        This method takes a tool call object from the AI model's response,
        executes the corresponding tool function, and returns a properly
        formatted message that can be added to the conversation history.
        
        Parameters
        ----------
        tool_call : Any
            Tool call object containing function execution details.
            Must have attributes:
            - function.name: Name of the function to call
            - function.arguments: JSON string of function arguments
            - id: Unique identifier for this tool call
        
        Returns
        -------
        Dict[str, str]
            Formatted tool response message containing:
            - tool_call_id: ID of the original tool call
            - role: Message role (always "tool")
            - name: Name of the executed function
            - content: Tool execution result as string
        
        Raises
        ------
        KeyError
            If the tool function is not registered with the agent
        json.JSONDecodeError
            If the function arguments are not valid JSON
        Exception
            If the tool function execution fails
        """

        function_name = tool_call.function.name
        function_to_call = self._tools[function_name]
        function_args = json.loads(tool_call.function.arguments)                
        function_response = str(function_to_call(**function_args))
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }


class _ReactMixin:
    """Mixin for handling ReAct-style tool calls with JSON format.
    
    This mixin provides methods for managing tool calls in structured JSON format,
    typical of ReAct-style approaches for AI agents. It handles the encoding
    of tool functions and execution of tool calls based on parsed JSON responses.
    
    The ReAct (Reasoning and Acting) pattern allows agents to reason about
    problems step-by-step and take actions using tools when needed.
    """
    
    def _encode_tool(self, func: Any) -> str:
        """Encode a function into a descriptive string format.
        
        This method creates a human-readable description of a tool function
        that can be included in prompts to help the AI model understand
        what tools are available and how to use them.
        
        Parameters
        ----------
        func : Any
            Function to encode. Must have attributes:
            - __name__: Function name
            - __doc__: Function documentation
        
        Returns
        -------
        str
            Descriptive string in the format:
            "function_name(signature): documentation"
            Newlines are replaced with spaces for single-line format.
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        encoded = func.__name__ + str(sig) + ": " + doc
        encoded = encoded.replace("\n", " ")
        return encoded
    
    def _call_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call in ReAct format.
        
        This method executes a tool call based on a structured dictionary
        containing the tool name and arguments. It's designed to work with
        the ReAct pattern where tools are called through JSON-formatted
        action specifications.
        
        Parameters
        ----------
        tool_call : Dict[str, Any]
            Tool call specification containing:
            - name: Name of the tool to call
            - arguments: Dictionary of tool arguments
        
        Returns
        -------
        Any
            Result of the tool execution
        
        Raises
        ------
        KeyError
            If the tool is not registered with the agent
        TypeError
            If the tool arguments don't match the function signature
        Exception
            If the tool execution fails
        """
        tool = self._tools[tool_call["name"]]
        kwargs = list(tool_call["arguments"].values())
        return tool(*kwargs)


class _AgenticLoop:
    """Base class for all agentic loop implementations.
    
    This class provides the core functionality for all AI agents, including
    tool management, message creation, model execution, and streaming support.
    It's designed to be extended by specific classes that implement different
    agentic approaches and reasoning patterns.
    
    The base class handles common operations like:
    - Tool registration and execution
    - Message formatting and conversation management
    - Model interaction (both streaming and non-streaming)
    - Step-based reasoning format parsing
    - Debug output and iteration limits
    
    Attributes
    ----------
    _model : Any
        AI model instance used for execution
    _agentic_prompt : str, optional
        Custom prompt for the agent (None for default)
    _debug : bool
        Flag to enable debug output
    _max_iter : Optional[int]
        Maximum number of iterations allowed (None for unlimited)
    _stream_callback : Optional[Callable[[str], None]]
        Callback function for handling streaming content
    _tools : Dict[str, Any]
        Dictionary of available tools, mapped by name
    """
    
    def __init__(self, model: Any, agentic_prompt: str=None, debug: bool=False, max_iter: Optional[int]=None, 
                 stream_callback: Optional[Callable[[str], None]]=None) -> None:
        """Initialize the agent with model and configuration.
        
        Parameters
        ----------
        model : Any
            AI model instance to use for execution
        agentic_prompt : str, optional
            Custom prompt for the agent (None to use default)
        debug : bool, default False
            Enable debug output and logging
        max_iter : Optional[int], default None
            Maximum number of iterations allowed (None for unlimited)
        stream_callback : Optional[Callable[[str], None]], default None
            Callback function for handling streaming content chunks
        """
        self._model = model
        self._agentic_prompt = agentic_prompt
        self._debug = debug
        self._max_iter = max_iter
        self._stream_callback = stream_callback
        self._tools = {}


    def register_tools(self, tools: List[Any]) -> None:
        """Register tools with the agent.
        
        Parameters
        ----------
        tools : List[Any]
            List of tool functions to register. Each tool must have a
            __name__ attribute for identification.
        """
        for tool in tools:
            self._tools[tool.__name__] = tool
    
    def enable_streaming(self, stream_callback: Optional[Callable[[str], None]] = None) -> None:
        """Enable streaming responses for this agent.
        
        When streaming is enabled, the agent will call the provided callback
        function with each content chunk as it's generated, allowing for
        real-time processing and display of the AI's response.
        
        Parameters
        ----------
        stream_callback : Optional[Callable[[str], None]], default None
            Callback function to handle streaming content chunks.
            If None, uses a default callback that prints content to console.
            The callback receives plain text content strings.
        """
        if stream_callback is None:
            def default_callback(content):
                print(content, end='', flush=True)
            
            self._stream_callback = default_callback
        else:
            self._stream_callback = stream_callback
    
    def disable_streaming(self) -> None:
        """Disable streaming responses for this agent.
        
        After calling this method, the agent will use standard (non-streaming)
        model execution for all subsequent requests.
        """
        self._stream_callback = None
    
    @classmethod
    def create_streaming(cls, model: Any, stream_callback: Optional[Callable[[str], None]] = None, 
                        **kwargs) -> '_AgenticLoop':
        """Create an agent instance with streaming enabled.
        
        This is a convenience class method that creates an agent instance
        with streaming already enabled, avoiding the need to call
        enable_streaming() separately.
        
        Parameters
        ----------
        model : Any
            AI model instance to use
        stream_callback : Optional[Callable[[str], None]], default None
            Callback function for handling streaming content chunks
        **kwargs
            Additional parameters to pass to the constructor
        
        Returns
        -------
        _AgenticLoop
            Agent instance with streaming enabled
        """
        return cls(model, stream_callback=stream_callback, **kwargs)

    def _get_tools(self) -> str:
        """Generate a descriptive string of available tools.
        
        This method creates a formatted string listing all registered tools
        with their signatures and documentation. This string is typically
        included in prompts to help the AI model understand what tools
        are available for use.
        
        Returns
        -------
        str
            Formatted string with descriptions of all available tools,
            one per line with " - " prefix. Returns empty string if no tools.
        """
        if not self._tools:
            return ""
        
        tools = []
        for tool_name, tool_func in self._tools.items():
            tools.append(f" - {self._encode_tool(tool_func)}")
        return "\n".join(tools)

    def _get_base_messages(self, agent_type: str, query: str) -> List[Dict[str, Any]]:
        """Generate base messages for the specific agent type.
        
        This method creates the initial message structure for the agent,
        including the appropriate prompt template and user query. The
        prompt template is selected based on the agent type and includes
        information about available tools.
        
        Parameters
        ----------
        agent_type : str
            Type of agent to determine which prompt template to use
        query : str
            User query to include in the prompt
        
        Returns
        -------
        List[Dict[str, Any]]
            List of base messages for the agent, including the prompt and query
        """
        tools = self._get_tools()
        prompt_id = (f"monoai/agents/prompts/{agent_type}.prompt" 
                    if self._agentic_prompt is None else self._agentic_prompt)
        
        prompt = Prompt(
            prompt_id=prompt_id,
            prompt_data={"query": query, "available_tools": tools}
        )
        
        return [prompt.as_dict()]

    def _debug_print(self, content: str) -> None:
        """Print debug information if debug mode is enabled.
        
        Parameters
        ----------
        content : str
            Content to print in debug mode
        """
        if self._debug:
            print(content)
            print("-------")
    
    def _parse_step_format(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the step-based format <STEP_TYPE>: <RESULT>.
        
        This method parses agent responses that follow the structured step format
        used by ReAct and similar reasoning patterns. It extracts the step type,
        step number, and content from formatted responses.
        
        Supported step types:
        - Thought: Reasoning and analysis steps
        - Action: Tool calls and actions (must be valid JSON)
        - Observation: Results from tool executions
        - Final answer: Conclusive responses to user queries
        
        Parameters
        ----------
        content : str
            Content to parse in the format <STEP_TYPE>: <RESULT>
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing:
            - step_type: Type of step (thought, action, observation, final answer)
            - step_number: Optional step number if present
            - content: Step content
            - action: Parsed action data (for action steps)
            - final_answer: Final answer content (for final answer steps)
            Returns None if parsing fails
        """
        if not content or not isinstance(content, str):
            return None
        
        content = content.strip()
        
        step_pattern = r'^(Thought|Action|Observation|Final answer)(?:\s+(\d+))?\s*:\s*(.*)$'
        match = re.match(step_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            step_type = match.group(1).lower()
            step_number = match.group(2) if match.group(2) else None
            step_content = match.group(3).strip()
            
            result = {
                "step_type": step_type,
                "step_number": step_number,
                "content": step_content
            }
            
            # Special handling for Action steps (must be JSON)
            if step_type == "action":
                try:
                    action_data = json.loads(step_content)
                    result["action"] = action_data
                except json.JSONDecodeError:
                    # If not valid JSON, keep content as raw string
                    result["action"] = {"raw": step_content}
            
            # Special handling for Final answer steps
            elif step_type == "final answer":
                result["final_answer"] = step_content
            
            return result
        
        return None

    def _execute_model_step(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a model step and return the response.
        
        This method handles both streaming and non-streaming model execution.
        If streaming is enabled and not already in progress, it uses the
        streaming method. Otherwise, it uses standard model execution.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            List of messages to send to the model
        
        Returns
        -------
        Dict[str, Any]
            Model response in standard OpenAI format
        """
        
        resp = self._model._execute(messages)
        return resp["choices"][0]["message"]
    
    def _execute_model_step_stream(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a model step with streaming and return the complete response.
        
        This method handles asynchronous streaming of model responses, collecting
        all chunks and building the final response. It properly manages async
        resources to avoid memory leaks and task warnings.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            List of messages to send to the model
        
        Returns
        -------
        Dict[str, Any]
            Complete model response in standard OpenAI format
        """

        import asyncio
        from litellm import stream_chunk_builder
        
        # Usa asyncio.run per gestire correttamente il loop
        async def run_streaming():
            chunks = []
            stream = None
            try:
                stream = self._model._execute_stream(messages)
                async for chunk in stream:
                    chunks.append(chunk)
                    content = chunk["choices"][0]["delta"]["content"]

                    if content is not None:
                        self._stream_callback(content)

            finally:
                if stream is not None:
                    try:
                        await stream.aclose()
                    except Exception:
                        pass
            
            return chunks

        try:
            chunks = asyncio.run(run_streaming())
            resp = stream_chunk_builder(chunks)
            return resp["choices"][0]["message"]
        except Exception as e:
            if self._debug:
                print(f"Streaming error: {e}, falling back to standard execution")
            # Fallback al metodo standard
            resp = self._model._execute(messages)
            return resp["choices"][0]["message"]
            
    def _create_base_response(self, query: str) -> Dict[str, Any]:
        """Create the base response structure.
        
        Parameters
        ----------
        query : str
            Original user query
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with base response structure:
            - prompt: Original user query
            - iterations: Empty list for iterations
        """
        return {"prompt": query, "iterations": []}

    def _handle_final_answer(self, iteration: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """Handle a final answer, returns True if this is the end.
        
        This method processes iterations that contain final answers and
        updates the response structure accordingly. It supports both
        the old format (final_answer key) and new step format.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration potentially containing a final answer
        response : Dict[str, Any]
            Response dictionary to update
        
        Returns
        -------
        bool
            True if a final answer was found and processed, False otherwise
        
        Notes
        -----
        This method modifies the response object directly.
        """
        if "final_answer" in iteration:
            response["iterations"].append(iteration)
            response["response"] = iteration["final_answer"]
            return True
        elif iteration.get("step_type") == "final answer":
            response["iterations"].append(iteration)
            response["response"] = iteration["content"]
            return True
        return False

    def _handle_tool_action(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Handle a tool action execution.
        
        This method processes iterations that contain tool actions, executes
        the corresponding tools, and updates the conversation with the results.
        It supports both old and new step-based formats.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration containing the tool action
        response : Dict[str, Any]
            Response dictionary to update
        messages : List[Dict[str, Any]]
            Message list to update with observation
        
        Notes
        -----
        This method modifies the response and messages objects directly.
        """
        if "action" in iteration and iteration["action"].get("name"):
            tool_call = iteration["action"]
            tool_result = self._call_tool(tool_call)
            iteration["observation"] = tool_result
            response["iterations"].append(iteration)
            
            msg = json.dumps({"observation": tool_result})
            messages.append({"type": "user", "content": msg})
        elif iteration.get("step_type") == "action" and "action" in iteration:
            tool_call = iteration["action"]
            tool_result = self._call_tool(tool_call)
            iteration["observation"] = tool_result
            response["iterations"].append(iteration)
            
            # Add observation in the new system format
            observation_msg = f"Observation {iteration.get('step_number', '')}: {tool_result}".strip()
            messages.append({"type": "user", "content": observation_msg})

    def _handle_default(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Handle default case for unhandled iterations.
        
        This method processes iterations that don't match specific handlers,
        adding them to the response and updating the conversation flow.
        It supports both step-based and JSON formats.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration to handle
        response : Dict[str, Any]
            Response dictionary to update
        messages : List[Dict[str, Any]]
            Message list to update
        
        Notes
        -----
        This method modifies the response and messages objects directly.
        """
        response["iterations"].append(iteration)
        
        # For new format, add content as user message
        if iteration.get("step_type") in ["thought", "observation"]:
            step_type = iteration["step_type"].capitalize()
            step_number = iteration.get("step_number", "")
            content = iteration["content"]
            message_content = f"{step_type} {step_number}: {content}".strip()
            messages.append({"type": "user", "content": message_content})
        else:
            # Fallback to JSON format for compatibility
            messages.append({"type": "user", "content": json.dumps(iteration)})

    def start(self, query: str) -> Dict[str, Any]:
        """Abstract method to start the agentic loop.
        
        This method must be implemented by subclasses to define the specific
        agentic behavior and reasoning pattern.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing iterations and final result
        
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError


class FunctionCallingAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    """Agent that uses OpenAI's native function calling.
    
    This agent implements a loop that leverages OpenAI's native function calling
    system, allowing the model to directly call available functions without
    manual response parsing. This approach is more reliable and efficient
    than text-based tool calling.
    
    The agent automatically handles:
    - Function call detection and execution
    - Tool result integration into conversation
    - Iteration limits to prevent infinite loops
    - Streaming support for real-time responses
    
    Attributes
    ----------
    _model : Any
        OpenAI model with function calling support
    _tools : Dict[str, Any]
        Available tools for the agent
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the agentic loop using function calling.
        
        This method processes user queries through OpenAI's function calling
        system, automatically executing tools when the model determines
        they are needed.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing:
            - prompt: Original user query
            - iterations: List of tool calls executed
            - response: Final model response
        """
        self._model._add_tools(list(self._tools.values()))
        messages = [{"type": "user", "content": query}]
        response = self._create_base_response(query)
        
        current_iter = 0
        max_iterations = self._max_iter if self._max_iter is not None else 10  # Limite di sicurezza
        
        while current_iter < max_iterations:
            
            if self._stream_callback is None:
                resp = self._execute_model_step(messages)
            else:
                resp = self._execute_model_step_stream(messages)

            messages.append(resp)
            content = resp["content"]
            self._debug_print(content)
            
            if resp.get("tool_calls"):
                for tool_call in resp["tool_calls"]:
                    tool_result = self._call_tool(tool_call)
                    response["iterations"].append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "result": tool_result["content"]
                    })
                    messages.append(tool_result)
            else:
                response["response"] = content
                break
            
            current_iter += 1
        
        # Se arriviamo qui, abbiamo raggiunto il limite di iterazioni
        if self._debug:
            print(f"Raggiunto limite di iterazioni ({max_iterations})")
        
        return response


class _BaseReactLoop(_AgenticLoop, _ReactMixin):
    """Base class for all ReAct-style agents.
    
    This class implements the standard loop for agents that use a ReAct-style
    approach, where the model produces structured JSON responses that are
    parsed and handled iteratively. The ReAct pattern combines reasoning
    and acting in a step-by-step manner.
    
    The base loop handles:
    - Step-based reasoning format parsing
    - Tool action execution and observation
    - Final answer detection
    - Custom iteration handlers
    - Error handling and fallbacks
    
    Attributes
    ----------
    _max_iter : Optional[int]
        Maximum number of iterations allowed
    """
    
    def _run_react_loop(self, query: str, agent_type: str, 
                        custom_handlers: Optional[Dict[str, callable]] = None) -> Dict[str, Any]:
        """Execute the standard ReAct loop.
        
        This method implements the core ReAct reasoning pattern where the
        agent alternates between thinking, acting, and observing until it
        reaches a final answer or hits iteration limits.
        
        Parameters
        ----------
        query : str
            User query to process
        agent_type : str
            Type of agent to determine which prompt template to use
        custom_handlers : Optional[Dict[str, callable]], optional
            Dictionary of custom handlers for specific iteration types.
            Keys are field names in the iteration, values are functions
            that handle those iterations.
        
        Returns
        -------
        Dict[str, Any]
            Agent response containing:
            - prompt: Original user query
            - iterations: List of processed iterations
            - response: Final response (if present)
        
        Notes
        -----
        This method automatically handles:
        - Final answers (final_answer)
        - Tool actions (action)
        - Custom cases via custom_handlers
        - Default cases for unhandled iterations
        - JSON error handling
        """
        messages = self._get_base_messages(agent_type, query)
        current_iter = 0
        response = self._create_base_response(query)
        
        # Handler personalizzati per casi speciali
        custom_handlers = custom_handlers or {}

        while True:
            if self._max_iter is not None and current_iter >= self._max_iter:
                break
            
            if self._stream_callback is None:
                resp = self._execute_model_step(messages)
            else:
                resp = self._execute_model_step_stream(messages)
            print(resp)
            messages.append(resp)
            content = resp["content"]

            self._debug_print(content)

            if content is not None:
                # Parsa il nuovo formato <TIPO DI STEP>: <RISULTATO>
                iteration = self._parse_step_format(content)

                if iteration:
                    # Gestione risposta finale
                    if self._handle_final_answer(iteration, response):
                        break
                    
                    # Gestione azioni di tool
                    if iteration["step_type"]=="action":
                        self._handle_tool_action(iteration, response, messages)
                        continue
                    
                    # Gestione casi personalizzati
                    handled = False
                    for key, handler in custom_handlers.items():
                        if iteration["step_type"] == key:
                            handler(iteration, response, messages)
                            handled = True
                            break
                    
                    if not handled:
                        self._handle_default(iteration, response, messages)
                else:
                    # Se non riesce a parsare, aggiungi come messaggio utente
                    messages.append({"type": "user", "content": content})

            current_iter += 1

        return response


class ReactAgenticLoop(_BaseReactLoop):
    """Standard ReAct agent.
    
    This agent implements the standard ReAct pattern, where the model
    produces JSON responses that are parsed and handled iteratively.
    The ReAct pattern combines reasoning and acting in a structured way.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the ReAct agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the ReAct loop
        """
        return self._run_react_loop(query, "react")


class ReactWithFCAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    """Agent that combines ReAct and Function Calling.
    
    This agent combines the ReAct approach with OpenAI's native function
    calling, allowing for hybrid tool call management. This provides
    the flexibility of ReAct reasoning with the reliability of function calling.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the hybrid ReAct + Function Calling agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response (to be implemented)
        
        Notes
        -----
        TODO: Implement combination of ReAct and Function Calling
        """
        # TODO: Implement combination of ReAct and Function Calling
        pass


class ProgrammaticAgenticLoop(_BaseReactLoop):
    """Programmatic agent.
    
    This agent implements a programmatic approach where the model
    produces code or structured instructions that are executed.
    It's designed for tasks that require code generation and execution.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the programmatic agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the programmatic loop
        """
        return self._run_react_loop(query, "programmatic")


class PlanAndExecuteAgenticLoop(_BaseReactLoop):
    """Plan-and-execute agent.
    
    This agent implements the plan-and-execute pattern, where the model
    first plans the actions and then executes them sequentially.
    This approach is useful for complex tasks that require careful planning.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the plan-and-execute agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the plan-and-execute loop
        """
        return self._run_react_loop(query, "plan_and_execute")


class ReflexionAgenticLoop(_BaseReactLoop):
    """Agent with reflection capabilities.
    
    This agent implements the reflexion pattern, where the model
    reflects on its own actions and decisions to improve performance.
    This self-reflective approach helps the agent learn from mistakes
    and improve its reasoning over time.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the reflexion agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the reflexion loop
        """
        return self._run_react_loop(query, "reflexion")


class SelfAskAgenticLoop(_BaseReactLoop):
    """Self-ask agent.
    
    This agent implements the self-ask pattern, where the model
    asks itself questions to guide the reasoning process.
    This approach helps break down complex problems into smaller,
    more manageable questions.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the self-ask agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the self-ask loop
        """
        return self._run_react_loop(query, "self_ask")


class SelfAskWithSearchLoop(_BaseReactLoop):
    """Self-ask agent with web search capabilities.
    
    This agent extends the self-ask pattern with the ability to
    perform web searches to obtain additional information.
    It's particularly useful for questions that require current
    or factual information not available in the model's training data.
    
    Attributes
    ----------
    _handle_search_query : callable
        Method for handling web search queries
    """
    
    def _handle_search_query(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Handle web search queries.
        
        This method processes search queries by executing web searches
        using the Tavily search engine and integrating the results
        into the conversation flow.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Current iteration containing the search query
        response : Dict[str, Any]
            Response dictionary to update
        messages : List[Dict[str, Any]]
            Message list to update with search results
        
        Notes
        -----
        This method modifies the response and messages objects directly.
        Uses the Tavily search engine for web searches.
        """
        from ..tools.websearch import search_web
        
        query = iteration["search_query"]
        result = search_web(query, engine="tavily")["text"]
        iteration["search_result"] = result
        
        msg = json.dumps({"query_results": result})
        messages.append({"type": "user", "content": msg})
        response["iterations"].append(iteration)
    
    def start(self, query: str) -> Dict[str, Any]:
        """Start the self-ask with search agentic loop.
        
        Parameters
        ----------
        query : str
            User query to process
        
        Returns
        -------
        Dict[str, Any]
            Agent response processed through the self-ask with search loop
        
        Notes
        -----
        This agent uses a custom handler for web search queries, allowing
        the model to obtain up-to-date information during the process.
        """
        custom_handlers = {"search_query": self._handle_search_query}
        return self._run_react_loop(query, "self_ask_with_search", custom_handlers)

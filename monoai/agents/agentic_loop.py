import json
import inspect
from typing import Any, Dict, List, Optional
from ..prompts import Prompt

class _FunctionCallingMixin:
    
    def _call_tool(self, tool_call: Any) -> Dict[str, str]:
        function_name = tool_call.function.name
        function_to_call = self._available_tools[function_name]
        function_args = json.loads(tool_call.function.arguments)                
        function_response = str(function_to_call(**function_args))
        return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }

class _ReactMixin:
    
    def _encode_tool(self, func: Any) -> str:
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        encoded = func.__name__+str(sig)+": "+doc
        encoded = encoded.replace("\n"," ")
        return encoded
    
    def _call_tool(self, tool_call: Dict[str, Any]) -> Any:
        tool = self._available_tools[tool_call["name"]]
        kwargs = list(tool_call["arguments"].values())
        return tool(*kwargs)

class _AgenticLoop:
    
    def __init__(self, model: Any, available_tools: List[Any], debug: bool, max_iter: Optional[int]) -> None:
        self._model = model
        self._debug = debug
        self._max_iter = max_iter
        
        self._available_tools = {}
        
        for tool in available_tools:
            self._available_tools[tool.__name__]=tool

    
    def start(self, query: str) -> Dict[str, Any]:
        pass
    

class FunctionCallingAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    
    def start(self, query: str) -> Dict[str, Any]:
        
        self._model._add_tools(list(self._available_tools.values()))
        messages = [{"type":"user", "content":query}]
        response = {"prompt":query, "iterations":[]} 
        
        while True:

            resp = self._model._execute(messages)
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if self._debug:
                print(content)
                print("-------")

            if content is not None:
                response["response"] = content
                break
            
            if resp["tool_calls"] is not None:
                tool_calls = resp["tool_calls"]
                for tool_call in tool_calls:
                    tool_result = self._call_tool(tool_call)
                    response["iterations"].append({
                        "name":tool_call.function.name , 
                        "arguments":tool_call.function.arguments,
                        "result":tool_result["content"]})
                    messages.append(tool_result)
        
        return response


class ReactAgenticLoop(_AgenticLoop, _ReactMixin):
        
    def start(self, query: str) -> Dict[str, Any]:

        tools = ""
        if self._available_tools != None:

            for tool in self._available_tools:
                tools+=" - "+self._encode_tool(self._available_tools[tool])+"\n"
                
        prompt = Prompt(prompt_id="monoai/agents/prompts/react.prompt", 
                        prompt_data={"query":query, 
                                     "available_tools":tools
                                     })
        
        messages = [prompt.as_dict()]
        
        current_iter = 0
        response = {"prompt":prompt, "iterations":[]} 

        while True:

            if self._max_iter is not None and current_iter>=self._max_iter:
                break
            
            resp = self._model._execute(messages)
            
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if self._debug:
                print(content)
                print("-------")

            if content is not None:
                iteration = json.loads(content)
                if "final_answer" in iteration:
                    response["iterations"].append(iteration)
                    response["response"] = iteration["final_answer"]
                    break
                elif "action" in iteration:
                    tool_call = iteration["action"]
                    tool_result = self._call_tool(tool_call)
                    iteration["observation"]=tool_result
                    response["iterations"].append(iteration)
                    msg = json.dumps({"observation":tool_result})
                    messages.append({"type":"user","content":msg})

            current_iter+=1

        return response
    

class ReactWithFCAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    
    def start(self, query: str) -> Dict[str, Any]:
        pass


class ProgrammaticAgenticLoop(_AgenticLoop, _ReactMixin):

    def start(self, query: str) -> Dict[str, Any]:
        tools = ""
        if self._available_tools != None:

            for tool in self._available_tools:
                tools+=" - "+self._encode_tool(self._available_tools[tool])+"\n"
                
        prompt = Prompt(prompt_id="monoai/agents/prompts/programmatic.prompt", 
                        prompt_data={"query":query, 
                                     "available_tools":tools
                                     })
        
        messages = [prompt.as_dict()]
        
        current_iter = 0
        response = {"prompt":prompt, "iterations":[]} 

        while True:

            if self._max_iter is not None and current_iter>=self._max_iter:
                break
            
            resp = self._model._execute(messages)
            
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if self._debug:
                print(content)
                print("-------")

            if content is not None:
                print(content)
                iteration = json.loads(content)
                if "final_answer" in iteration:
                    response["iterations"].append(iteration)
                    response["response"] = iteration["final_answer"]
                    break
                elif "action" in iteration and iteration["action"]["name"] is not None:
                    tool_call = iteration["action"]
                    tool_result = self._call_tool(tool_call)
                    iteration["observation"]=tool_result
                    response["iterations"].append(iteration)
                    msg = json.dumps({"observation":tool_result})
                    messages.append({"type":"user","content":msg})
                else:
                    messages.append({"type":"user","content":content})

            current_iter+=1

        return response


class PlanAndExecuteAgenticLoop(_AgenticLoop, _ReactMixin):


    def start(self, query: str) -> Dict[str, Any]:
        tools = ""
        if self._available_tools != None:

            for tool in self._available_tools:
                tools+=" - "+self._encode_tool(self._available_tools[tool])+"\n"
                
        prompt = Prompt(prompt_id="monoai/agents/prompts/plan_and_execute.prompt", 
                        prompt_data={"query":query, 
                                     "available_tools":tools
                                     })
        
        messages = [prompt.as_dict()]
        
        current_iter = 0
        response = {"prompt":prompt, "iterations":[]} 

        while True:

            if self._max_iter is not None and current_iter>=self._max_iter:
                break
            
            resp = self._model._execute(messages)
            
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if self._debug:
                print(content)
                print("-------")

            if content is not None:
                print(content)
                iteration = json.loads(content)
                if "final_answer" in iteration:
                    response["iterations"].append(iteration)
                    response["response"] = iteration["final_answer"]
                    break
                elif "action" in iteration and iteration["action"]["name"] is not None:
                    tool_call = iteration["action"]
                    tool_result = self._call_tool(tool_call)
                    iteration["observation"]=tool_result
                    response["iterations"].append(iteration)
                    msg = json.dumps({"observation":tool_result})
                    messages.append({"type":"user","content":msg})
                else:
                    messages.append({"type":"user","content":content})

            current_iter+=1

        return response


class ReflexionAgenticLoop(_AgenticLoop, _ReactMixin):

    def start(self, query: str) -> Dict[str, Any]:
        tools = ""
        if self._available_tools != None:

            for tool in self._available_tools:
                tools+=" - "+self._encode_tool(self._available_tools[tool])+"\n"
                
        prompt = Prompt(prompt_id="monoai/agents/prompts/reflexion.prompt", 
                        prompt_data={"query":query, 
                                     "available_tools":tools
                                     })
        
        messages = [prompt.as_dict()]
        
        current_iter = 0
        response = {"iterations":[]} 

        while True:

            if self._max_iter is not None and current_iter>=self._max_iter:
                break
            
            resp = self._model._execute(messages)
            
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if self._debug:
                print(content)
                print("-------")

            if content is not None:
                iteration = json.loads(content)
                if "final_answer" in iteration:
                    response["iterations"].append(iteration)
                    response["response"] = iteration["final_answer"]
                    break
                elif "action" in iteration:
                    tool_call = iteration["action"]
                    tool_result = self._call_tool(tool_call)
                    iteration["observation"]=tool_result
                    iteration["observation"]=tool_result
                    response["iterations"].append(iteration)
                    msg = json.dumps({"observation":tool_result})
                    messages.append({"type":"user","content":msg})
                else:
                    response["iterations"].append(iteration)
                    messages.append({"type":"user","content":content})

            current_iter+=1

        return response
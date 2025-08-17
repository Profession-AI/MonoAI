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
    
    def __init__(self, model: Any, available_tools: List[Any], agentic_prompt:str, debug: bool, max_iter: Optional[int]) -> None:
        self._model = model
        self._agentic_prompt = agentic_prompt
        self._debug = debug
        self._max_iter = max_iter
        
        self._available_tools = {}
        
        for tool in available_tools:
            self._available_tools[tool.__name__]=tool

    def _get_tools(self)->str:
        tools = ""
        if self._available_tools != None:

            for tool in self._available_tools:
                tools+=" - "+self._encode_tool(self._available_tools[tool])+"\n"
        return tools


    def _get_base_messages(self, agent_type:str, query:str)->str:
        tools = self._get_tools()
        prompt = Prompt(prompt_id=f"monoai/agents/prompts/{agent_type}"+".prompt" if self._agentic_prompt is None else self._agentic_prompt, 
                        prompt_data={"query":query, 
                                     "available_tools":tools
                                     })
        
        messages = [prompt.as_dict()]
        return messages

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
        
        messages = self._get_base_messages("react", query)
        
        current_iter = 0
        response = {"prompt":query, "iterations":[]} 

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

        messages = self._get_base_messages("programmatic", query)        
        
        current_iter = 0
        response = {"prompt":query, "iterations":[]} 

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

        messages = self._get_base_messages("plan_and_execute", query)          
        
        current_iter = 0
        response = {"prompt":query, "iterations":[]} 

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
        
        messages = self._get_base_messages("reflexion", query)  
        
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
 

class SelfAskAgenticLoop(_AgenticLoop, _ReactMixin):

    def start(self, query: str) -> Dict[str, Any]:
        
        messages = self._get_base_messages("self_ask", query)  
        
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
                else:
                    response["iterations"].append(iteration)
                    messages.append({"type":"user","content":content})

            current_iter+=1

        return response


class SelfAskWithSearchLoop(_AgenticLoop, _ReactMixin):

    def start(self, query: str) -> Dict[str, Any]:
        
        messages = self._get_base_messages("self_ask_with_search", query)  
        
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
                elif "search_query" in iteration and iteration["search_query"] is not None:
                    from ..tools.websearch import search_web
                    
                    query = iteration["search_query"] 
                    result = search_web(query, engine="tavily")["text"]
                    iteration["search_result"] = result
                    result = json.dumps({"query_results":result})
                    messages.append({"type":"user", "content":result})
                    response["iterations"].append(iteration)
                else:
                    response["iterations"].append(iteration)
                    messages.append({"type":"user","content":content})

            current_iter+=1

        return response

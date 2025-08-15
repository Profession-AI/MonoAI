import json
from ..prompts import Prompt

class _AgenticLoop:
    
    def __init__(self, model, available_tools, debug, max_iter):
        self._model = model
        self._available_tools = available_tools
        self._debug = debug
        self._max_iter = max_iter
    
    def _call_tool(self, tool_call):
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

    
    def start(self):
        pass
    

class FunctionCallingAgenticLoop(_AgenticLoop):
    
    def start(self, prompt):
        
        messages = [{"type":"user", "content":prompt}] 
        
        while True:

            resp = self._model._execute(messages)
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if self._debug:
                print(content)
                print("-------")

            if content is not None:
                break
            
            if resp["tool_calls"] is not None:
                tool_calls = resp["tool_calls"]
                for tool_call in tool_calls:
                    tool_result = self._call_tool(tool_call)
                    messages.append(tool_result)
        
        return content


class ReactAgenticLoop(_AgenticLoop):
    
    def start(self, query):

        prompt = Prompt(prompt_id="monoai/agents/prompts/react.prompt", prompt_data={"query":query})
        messages = [prompt.as_dict()]
        print(messages)
        
        current_iter = 0

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
                content_json = json.loads(content)
                if "final_answer" in content_json:
                    break
            
            if resp["tool_calls"] is not None:
                tool_calls = resp["tool_calls"]
                for tool_call in tool_calls:
                    tool_result = self._call_tool(tool_call)
                    messages.append(tool_result)

            current_iter+=1

        return content
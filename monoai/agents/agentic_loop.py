import json

class _AgenticLoop:
    
    def __init__(self, model, available_tools):
        self._model = model
        self._available_tools = available_tools
    
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
    

class FunctionCallAgenticLoop(_AgenticLoop):
    
    def start(self, prompt):
        
        messages = [{"type":"user", "content":prompt}] 
        
        while True:

            resp = self._model._execute(messages)
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if content is not None:
                break
            else:
                if resp["tool_calls"] is not None:
                    tool_calls = resp["tool_calls"]
                    for tool_call in tool_calls:
                        tool_result = self._call_tool(tool_call)
                        messages.append(tool_result)
        
        return content


class ReactAgenticLoop:
    
    def run(self):
        pass
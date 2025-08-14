from ..models import Model
import json

class Agent():

    def __init__(self, model:Model, tools):
        self._model = model
        self._model._add_tools(tools)
        self._available_tools = {}
        for tool in tools:
            self._available_tools[tool.__name__]=tool

    def run(self, prompt: str):
        return self._loop(prompt)


    def _loop(self, prompt:str):

        finish = False
        messages = [{"role": "user", "content": prompt}]
        while not finish:

            resp = self._model._execute(messages)
            print(resp)
            resp = resp["choices"][0]["message"]
            messages.append(resp)
            content = resp["content"]

            if content is not None:
                finish=True
            else:
                if resp["tool_calls"] is not None:
                    tool_calls = resp["tool_calls"]
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = self._available_tools[function_name]
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = function_to_call(
                            location=function_args.get("location"),
                            unit=function_args.get("unit"),
                        )
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
        
        return content




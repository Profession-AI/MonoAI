from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool


class WebSearch:

    def __init__(self, engine: str = "duckduckgo"):

        if engine == "duckduckgo":
            self._engine = duckduckgo_search_tool
        elif engine == "tavily":
            self._engine = tavily_search_tool
        else:
            raise ValueError(f"Invalid engine: {engine} (must be 'duckduckgo' or 'tavily')")

    def get_tool(self):
        return self._engine()

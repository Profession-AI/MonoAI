from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool
from ._base_tools import BaseTool

class WebSearch(BaseTool):

    def __init__(self, engine: str = "duckduckgo"):

        if engine == "duckduckgo":
            self._tool= duckduckgo_search_tool
        elif engine == "tavily":
            self._tool = tavily_search_tool
        else:
            raise ValueError(f"Invalid engine: {engine} (must be 'duckduckgo' or 'tavily')")

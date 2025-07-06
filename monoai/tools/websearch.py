class WebSearch():

    """
    WebSearch is a tool that allows you to search the web. 
    It uses the DuckDuckGo or Tavily API to search the web.

    Examples
    --------

    ```python
    websearch = WebSearch(engine="duckduckgo", max_results=5)
    result = websearch.search("What is the capital of France?")
    print(result["text"]) # print the result merged into a single string
    print(result["data"]) # print the result as a list of dictionaries
    ```
    """

    def __init__(self, engine: str = "duckduckgo", max_results: int = 5, exclude_domains: list[str] = None):

        """
        Initialize the WebSearch tool.

        Parameters:
        ----------
        engine: str, optional
            The search engine to use (duckduckgo or tavily, default is duckduckgo)
        max_results: int, optional
            The maximum number of results to return (default is 5)
        exclude_domains: list[str], optional
            The domains to exclude from the search (default is None)
        """

        if engine == "duckduckgo":
            self._engine = _DuckDuckGoSearch(max_results, exclude_domains)
        elif engine == "tavily":
            self._engine = _TavilySearch(max_results, exclude_domains)
        else:
            raise ValueError(f"Invalid engine: {engine} (must be 'duckduckgo' or 'tavily')")

    def search(self, query: str):

        """
        Search the web.

        Parameters:
        ----------
        query: str
            The query to search for

        Returns:
        -------
        dict
            The text response from the search engine.
        """
        response, text_response = self._engine.search(query)
        return {"data": response, "text": text_response}
    
from duckduckgo_search import DDGS

class _BaseSearch():

    def __init__(self, max_results: int = 5, exclude_domains: list[str] = None):
        self._max_results = max_results
        self._exclude_domains = exclude_domains

    def search(self, query: str):
        pass

    def _post_process(self, response: list[dict], title_key: str = "title", text_key: str = "body", url_key: str = "url"):
        response = [{"title": item[title_key], "text": item[text_key], "url": item[url_key]} for item in response]
        text_response = "\n\n".join([item["title"] + "\n\n" + item["text"] for item in response])
        return response, text_response

class _DuckDuckGoSearch(_BaseSearch):

    def __init__(self, max_results: int = 5, exclude_domains: list[str] = None):
        super().__init__(max_results, exclude_domains)
        self._client = DDGS()

    def search(self, query: str):
        response = self._client.text(query, max_results=self._max_results)
        return self._post_process(response, title_key="title", text_key="body", url_key="href")
    

from tavily import TavilyClient
from monoai.keys.keys_manager import load_key

class _TavilySearch(_BaseSearch):

    def __init__(self, max_results: int = 5, exclude_domains: list[str] = None):
        super().__init__(max_results, exclude_domains)
        load_key("tavily")
        self._client = TavilyClient()

    def search(self, query: str):
        response = self._client.search(query, max_results=self._max_results, exclude_domains=self._exclude_domains)
        response = response["results"]
        return self._post_process(response, title_key="title", text_key="content", url_key="url")

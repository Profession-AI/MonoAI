from coicoi.tools.websearch import WebSearch

websearch = WebSearch(engine="tavily", max_results=5, exclude_domains=["https://www.biglucainternational.com"])
result = websearch.search("https://www.biglucainternational.com")
print(result)








from coicoi.models import Model
from coicoi.prompts import Prompt
from coicoi.tools import WebScraping

websearch = WebScraping(engine="selenium")
result = websearch.scrape("https://www.scrapethissite.com/pages/simple/")
#print(result)
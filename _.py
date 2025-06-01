from coicoi.models import Model
from coicoi.prompts import Prompt
<<<<<<< HEAD
prompt = Prompt(prompt="2+2=", response_type="int")
model = Model()
print(model.ask(prompt))
=======
from coicoi.tools import WebScraping

websearch = WebScraping(engine="selenium")
result = websearch.scrape("https://www.scrapethissite.com/pages/simple/")
#print(result)
>>>>>>> 8d1c9aebcf9ef3bd2dd163bd9b470fc01fddea17

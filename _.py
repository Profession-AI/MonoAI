from coicoi.models import Model
from coicoi.prompts import IterativePrompt

model = Model()

chapters = ["variabili", "espressioni condizionali", "cicli", "funzioni"]
data = {"topic":"programmazione con python", "chapters":chapters}

prompt = IterativePrompt(prompt_id="test_iter", prompt_data=data, iter_data=chapters, retain_all=True)
print(model.ask(prompt))


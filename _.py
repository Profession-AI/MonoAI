from coicoi.models import Model
from coicoi.prompts import IterativePrompt


model = Model(
    provider="openai",
    model="gpt-4o-mini",
)

chapters = ["basi del prompting", "chain of thoughts", "prompt security"]
prompt = """
Genera il contenuto di un capitolo di un corso {topic}.
I capitoli sono {chapters}. Genera il contenuto del capitolo {{data}}
"""

prompt_memory = """
Assicurati che sia coerente con il contenuto del capitolo precedente: {{data}}
"""

prompt = IterativePrompt(prompt_id="test_iter", prompt_data={"topic":"prompt engineering", "chapters":chapters}, iter_data=chapters)
print(model.ask(prompt))

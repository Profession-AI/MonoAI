from models.model import Model
from prompts.prompt_chain import PromptChain

model = Model(
    provider_name="openai",
    model_name="gpt-4o-mini",
    count_tokens=True,
    count_cost=True)

chain = PromptChain([
    "What is the capital of France?",
    "Given the city name, tell me its population."
])

result = model.ask(chain)
print(result)

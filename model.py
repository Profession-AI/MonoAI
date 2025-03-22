from pydantic_ai import Agent
from keys_manager import load_keys

class Model:
    def __init__(self, provider_name: str, model_name: str):
        load_keys()
        self.provider_name = provider_name
        self.model_name = model_name
        self._agent = Agent(provider_name + ":" + model_name)

    def ask_async(self, question: str):
        return self._agent.run(question)

    def ask(self, question: str):
        return self._agent.run_sync(question)
    

if __name__ == "__main__":
    model = Model("openai", "gpt-4o-mini")
    print(model.ask("What is the capital of France?"))

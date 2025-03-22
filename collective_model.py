from multi_model import MultiModel
from model import Model

class CollectiveModel:
    def __init__(self, models: list[dict], aggreator: dict):
        self._multi_model = MultiModel(models)
        self._aggreator = Model(aggreator['provider'], aggreator['model'])

    def ask(self, question: str, aggregator_prompt: str = None):
        
        answers = self._multi_model.ask(question)
        
        aggregator_prompt = aggregator_prompt or "Please aggregate the following answers into a single answer: {answers}"
        answer = self._aggreator.ask(aggregator_prompt.format(answers=answers))
        return answer


if __name__ == "__main__":
    collective_model = CollectiveModel(
        models=[{'provider': 'openai', 'model': 'gpt-4o-mini'},
        {'provider': 'deepseek', 'model': 'deepseek-chat'},
    ], aggreator={'provider': 'deepseek', 'model': 'deepseek-chat'})

    answer = collective_model.ask("What is the capital of France?")
    print(answer)


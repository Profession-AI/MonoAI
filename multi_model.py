from model import Model
import asyncio


class MultiModel:


    def __init__(self, models: list[dict] ):        
        self._models = [Model(model['provider'], model['model']) for model in models]


    async def _task(self, model, question):
        answer = await model.ask_async(question)
        return (model.model_name, answer.data)
    

    async def _ask_async(self, question: str):
        tasks = [self._task(model, question) for model in self._models]
        results = await asyncio.gather(*tasks)
        answers = {}
        for model_name, answer in results:
            answers[model_name] = answer
        return answers

    def ask(self, question: str):
        return asyncio.run(self._ask_async(question))



if __name__ == "__main__":
    multi_model = MultiModel([
        {'provider': 'openai', 'model': 'gpt-4o-mini'},
        {'provider': 'deepseek', 'model': 'deepseek-chat'},
    ])

    print(multi_model.ask("What is the meaning of life?"))

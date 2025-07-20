from monoai.models import Model
from monoai.prompts import Prompt
from pydantic import BaseModel

class Response(BaseModel):
    result: float


model = Model(provider="mistral", model="mistral-tiny")
prompt = Prompt(prompt="3+2.5=", response_type=Response)
print(model.ask(prompt))
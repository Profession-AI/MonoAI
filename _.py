from coicoi.models import Model
from coicoi.prompts import Prompt
from pydantic import BaseModel

class Response(BaseModel):
    response: int

model = Model(
    provider="openai",
    model="gpt-4o-mini")

prompt = Prompt(prompt="2+2=", response_type=Response)

print(model.ask(prompt))


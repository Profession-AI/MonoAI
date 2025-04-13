from coicoi.models import Model
from coicoi.models._keys_manager import _KeyManager


model = Model(
    provider="openai",
    model="gpt-4o-mini",
)

print(model.ask("2+2"))

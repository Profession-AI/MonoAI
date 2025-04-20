from openai import OpenAI
from ._keys_manager import load_key 

class ImageModel():
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

        if provider.lower() != "openai":
            raise ValueError(f"Provider {provider} not supported")
        if model.lower() != "dall-e-3":
            raise ValueError(f"Model {model} not supported")
        load_key(provider)
        self._client = OpenAI()

    def generate(self, prompt: str, size: str = "1024x1024", quality: str = "standard", n: int = 1):
        response = self._client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        return response




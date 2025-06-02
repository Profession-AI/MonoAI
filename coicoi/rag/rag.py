from litellm import embedding
from coicoi.keys.keys_manager import load_key

class RAG:

    def __init__(self, 
                 provider: str,
                 vectorizer: str, 
                 database: str):
        
        load_key(provider)
        self._vectorizer = vectorizer
        self._db = database


    def embed(self, text: str):
        result = embedding(
            model=self._vectorizer,
            input=text
        )
        return result

    

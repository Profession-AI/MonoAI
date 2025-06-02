import chromadb
from litellm import embedding

class BaseVectorDB:

    def __init__(self, name: str, vectorizer_provider: str, vectorizer_model: str):
        self._name = name
        self._vectorizer_provider = vectorizer_provider
        self._vectorizer_model = vectorizer_model

    def _embed(self, documents: list[str]):
        result = embedding(
            model=self._vectorizer_model,
            input=documents
        )
        return result

    def add(self, documents: list[str]):
        pass

    def query(self, query: str):
        pass
    
    def delete(self):
        pass
    

class ChromaVectorDB(BaseVectorDB):
    
    

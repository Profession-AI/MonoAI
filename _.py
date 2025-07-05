from coicoi.rag.rag import RAG
from coicoi.rag.vectordb import ChromaVectorDB
from coicoi.models.model import Model
from coicoi.rag.documents_builder import DocumentsBuilder


documents_builder = DocumentsBuilder(chunk_size=500, chunk_overlap=100)
documents, metadatas, ids = documents_builder.from_url("https://www.profession.ai")

vector_db = ChromaVectorDB(name="coicoi")
vector_db.add(documents, metadatas, ids)

model = Model(provider="openai", model="gpt-4o-mini")
model._add_rag(RAG(database="coicoi", vector_db="chroma"))
result = model.ask("Che tipo di certificazione offre ProfessionAI?")
print(result)

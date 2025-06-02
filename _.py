from coicoi.rag import RAG

rag = RAG("openai", "text-embedding-ada-002", "sqlite")

embedding = rag.embed(["Le lezioni live sono il Giovedì alle 14", 
                       "Il corso di Python inizia a Settembre",
                       "I docenti non fanno consulenze private"])
print(embedding)
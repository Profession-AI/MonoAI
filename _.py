from monoai.rag.rag import RAG
from monoai.rag.vectordb import ChromaVectorDB
from monoai.models.model import Model
from monoai.rag.documents_builder import DocumentsBuilder

s = """
Spett.le Sig. Emiliano Ruozzo
con la presente desideriamo manifestare il nostro interesse nei Suoi confronti in relazione all’avvio di una collaborazione professionale presso la
nostra azienda, per il ruolo di Chief Commercial Officer.
Desideriamo anticipare i principali termini e condizioni che intendiamo proporLe.
Tipologia contrattuale:
- Contratto di lavoro subordinato a tempo indeterminato, disciplinato dal CCNL Commercio – livello 2
Retribuzione:
- Retribuzione Annua Lorda (RAL): €34.120,00, corrisposta in 14 mensilità
- Riconoscimento di una commissione pari al 5% sull’importo lordo effettivamente incassato dall’azienda per le vendite generate, senza alcuna
deduzione di costi, spese o oneri relativi all’erogazione del servizio, secondo modalità e tempistiche che saranno dettagliate nel contratto di lavoro.
- Assegnazione di una quota pari allo 0,25% del capitale sociale (“work for equity”), subordinatamente alle condizioni e alle modalità che saranno
definite nel relativo accordo.
Benefit:
- Dotazione di computer e smartphone aziendale
- Polizza sanitaria integrativa Unisalute
- Buoni pasto Edenred per un valore di €150 mensili
- Possibilità di svolgere la prestazione lavorativa in modalità smart working o in co-working
- Ferie illimitate retribuite, da concordare con l’azienda nel rispetto delle esigenze organizzative
Si precisa che la presente comunicazione non costituisce proposta contrattuale né impegno vincolante tra le parti, ma rappresenta esclusivamente
una manifestazione di interesse, finalizzata all’approfondimento di una possibile collaborazione professionale alle condizioni sopra indicate, fatte salve
le verifiche di rito e la reciproca volontà di formalizzare un successivo accordo.
"""

builder = DocumentsBuilder(chunk_strategy="semantic", chunk_size=10)
documents, metadatas, ids = builder.from_str(s)
print(documents)

"""
model = Model(provider="openai", model="gpt-4o-mini")
model._add_rag(RAG(database="monoai", vector_db="chroma"))
result = model.ask("Qual è il valore nominale del piano?")
print(result)
"""
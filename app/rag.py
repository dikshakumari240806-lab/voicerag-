import time
from .config import settings
from .embeddings import Embedder
from .vectorstore import VectorStore
from .guardrails import clean_query,retrieval_guard,grounding_guard
from .llm import LLMClient
from .models import Answer

class RAG:
    def __init__(self,backend=None):
        self.embedder=Embedder()
        self.store=VectorStore(backend or settings.vector_backend,self.embedder.dimension)
        self.llm=LLMClient()
    def ingest(self,directory="data",strategy="all"):
        from .loaders import load_directory
        from .chunking import make_chunks
        docs=load_directory(directory); chunks=[]
        for d in docs: chunks+=make_chunks(d,strategy,self.embedder)
        if not chunks: raise RuntimeError("No documents in data/")
        self.store.add(chunks,self.embedder.encode([c.text for c in chunks]))
        return {"documents":len(docs),"chunks":len(chunks),"strategy":strategy,"backend":self.store.backend}
    def ask(self,question,transcript=""):
        t0=time.perf_counter(); ok,q,reason=clean_query(question)
        if not ok:return Answer("",[],False,0,{"total_ms":(time.perf_counter()-t0)*1000},transcript,reason)
        v=self.embedder.encode([q])[0]; tr=time.perf_counter(); results=self.store.search(v,settings.top_k); retrieval_ms=(time.perf_counter()-tr)*1000
        ok,reason=retrieval_guard(results)
        if not ok:return Answer("I don't have enough information in the provided documents to answer this reliably.",[],False,0,{"retrieval_ms":retrieval_ms,"total_ms":(time.perf_counter()-t0)*1000},transcript,reason)
        tg=time.perf_counter(); answer,context=self.llm.answer(q,results); generation_ms=(time.perf_counter()-tg)*1000
        ok,reason=grounding_guard(answer,context)
        if not ok:return Answer("I don't have enough information in the provided documents to answer this reliably.",[],False,0,{"retrieval_ms":retrieval_ms,"generation_ms":generation_ms,"total_ms":(time.perf_counter()-t0)*1000},transcript,reason)
        best=max(r.score for r in results)
        sources=[{"source":r.chunk.source,"strategy":r.chunk.strategy,"score":round(r.score,4),"chunk_id":r.chunk.id} for r in results]
        return Answer(answer,sources,True,min(1,max(0,best)),{"retrieval_ms":retrieval_ms,"generation_ms":generation_ms,"total_ms":(time.perf_counter()-t0)*1000},transcript)

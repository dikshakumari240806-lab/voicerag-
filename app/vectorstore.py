import json
from pathlib import Path
import numpy as np, faiss
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct
from .config import settings
from .models import Chunk,Retrieval

class VectorStore:
    def __init__(self,backend,dimension):
        self.backend=backend
        if backend=="faiss":
            self.dir=Path(settings.faiss_dir); self.dir.mkdir(parents=True,exist_ok=True)
            self.ip=self.dir/"index.faiss"; self.meta=self.dir/"metadata.json"
            self.index=faiss.read_index(str(self.ip)) if self.ip.exists() else faiss.IndexFlatIP(dimension)
            self.chunks=[Chunk(**x) for x in json.loads(self.meta.read_text())] if self.meta.exists() else []
        elif backend=="qdrant":
            self.client=QdrantClient(path=settings.qdrant_path)
            self.collection="voice_rag"
            try: self.client.get_collection(self.collection)
            except Exception:
                self.client.create_collection(collection_name=self.collection,vectors_config=VectorParams(size=dimension,distance=Distance.COSINE))
        else: raise ValueError("backend must be faiss or qdrant")

    def add(self,chunks,vectors):
        vectors=np.asarray(vectors,dtype="float32")
        if self.backend=="faiss":
            self.index.add(vectors); self.chunks.extend(chunks)
            faiss.write_index(self.index,str(self.ip))
            self.meta.write_text(json.dumps([c.__dict__ for c in self.chunks],ensure_ascii=False,indent=2))
        else:
            import hashlib
            points=[]
            for c,v in zip(chunks,vectors):
                pid=int(hashlib.sha1(c.id.encode()).hexdigest()[:15],16)
                points.append(PointStruct(id=pid,vector=v.tolist(),payload=c.__dict__))
            self.client.upsert(collection_name=self.collection,points=points)

    def search(self,vector,k):
        q=np.asarray(vector,dtype="float32").reshape(1,-1)
        if self.backend=="faiss":
            if self.index.ntotal==0:return []
            scores,ids=self.index.search(q,min(k,self.index.ntotal))
            return [Retrieval(self.chunks[int(i)],float(s)) for s,i in zip(scores[0],ids[0]) if i>=0]
        hits=self.client.query_points(collection_name=self.collection,query=q[0].tolist(),limit=k).points
        return [Retrieval(Chunk(**h.payload),float(h.score)) for h in hits]

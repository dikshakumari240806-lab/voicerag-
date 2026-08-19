from .loaders import load_directory
from .chunking import make_chunks
from .embeddings import Embedder

for d in load_directory("data"):
    e=Embedder()
    for s in ["fixed","recursive","semantic"]:
        c=make_chunks(d,s,e)
        print(s, "chunks=",len(c), "avg_chars=",round(sum(len(x.text) for x in c)/max(1,len(c)),1))

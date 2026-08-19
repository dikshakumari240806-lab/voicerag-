import json,statistics
from pathlib import Path
from .rag import RAG

QUERIES=["What is the scholarship deadline?","Who is eligible for the scholarship?","What documents should applicants prepare?","How are applications evaluated?","Where should applicants go for application issues?","What is the capital of France?","Tell me something unrelated to the scholarship?"]

def pct(xs,p):
    xs=sorted(xs); k=(len(xs)-1)*p/100; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def run(backend="faiss",repeats=2):
    rag=RAG(backend); rows=[]
    for q in QUERIES:
        for _ in range(repeats):
            a=rag.ask(q); rows.append({"query":q,**a.to_dict()})
    vals=[r["latency_ms"]["total_ms"] for r in rows]
    summary={"n":len(vals),"p50_ms":round(pct(vals,50),2),"p70_ms":round(pct(vals,70),2),"p100_ms":round(max(vals),2),"mean_ms":round(statistics.mean(vals),2)}
    out={"summary":summary,"rows":rows}; Path("benchmarks").mkdir(exist_ok=True); Path("benchmarks/results.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)); return out

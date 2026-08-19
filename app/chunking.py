import re, uuid
from .models import Chunk

def fixed(doc,size=1400,overlap=250):
    out=[]; start=0; idx=0
    while start<len(doc.text):
        end=min(len(doc.text),start+size); text=doc.text[start:end].strip()
        if text: out.append(Chunk(str(uuid.uuid4()),doc.source,text,"fixed",idx,{"start":start,"end":end})); idx+=1
        if end==len(doc.text): break
        start=end-overlap
    return out

def recursive(doc,max_chars=1400):
    paras=[p.strip() for p in re.split(r"\n\s*\n",doc.text) if p.strip()]
    out=[]; current=""; idx=0
    for para in paras:
        if len(current)+len(para)+2<=max_chars:
            current=(current+"\n\n"+para).strip()
        else:
            if current: out.append(Chunk(str(uuid.uuid4()),doc.source,current,"recursive",idx,{})); idx+=1
            for s in re.split(r"(?<=[.!?])\s+",para):
                if len(current)+len(s)+1<=max_chars: current=(current+" "+s).strip()
                else:
                    if current: out.append(Chunk(str(uuid.uuid4()),doc.source,current,"recursive",idx,{})); idx+=1
                    current=s
    if current: out.append(Chunk(str(uuid.uuid4()),doc.source,current,"recursive",idx,{}))
    return out

def semantic(doc,embedder,threshold=.42,max_chars=1600):
    sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+",doc.text.replace("\n"," ")) if s.strip()]
    if not sentences: return []
    vecs=embedder.encode(sentences); out=[]; current=sentences[0]; idx=0
    for i in range(1,len(sentences)):
        sim=float(vecs[i-1]@vecs[i]); candidate=current+" "+sentences[i]
        if sim>=threshold and len(candidate)<=max_chars: current=candidate
        else:
            out.append(Chunk(str(uuid.uuid4()),doc.source,current,"semantic",idx,{"similarity":sim})); idx+=1; current=sentences[i]
    out.append(Chunk(str(uuid.uuid4()),doc.source,current,"semantic",idx,{}))
    return out

def make_chunks(doc,strategy,embedder):
    if strategy=="fixed": return fixed(doc)
    if strategy=="recursive": return recursive(doc)
    if strategy=="semantic": return semantic(doc,embedder)
    if strategy=="all": return fixed(doc)+recursive(doc)+semantic(doc,embedder)
    raise ValueError("Use fixed, recursive, semantic, or all")

import re
from .config import settings

def clean_query(q):
    q=re.sub(r"\s+"," ",q or "").strip()
    if not q:return False,"","Empty query"
    if len(q)>1000:return False,"","Query too long"
    return True,q,None

def retrieval_guard(results):
    if not results:return False,"No relevant documents found"
    best=max(r.score for r in results)
    if best<settings.min_retrieval_score:
        return False,f"Best retrieval score {best:.3f} is below threshold {settings.min_retrieval_score:.3f}"
    return True,None

def grounding_guard(answer,context):
    if not answer.strip():return False,"Empty model answer"
    if "I don't have enough information" in answer:return True,None
    a=set(re.findall(r"[a-zA-Z0-9]{5,}",answer.lower()))
    c=set(re.findall(r"[a-zA-Z0-9]{5,}",context.lower()))
    return (len(a&c)>=2,"Insufficient evidence in retrieved context")

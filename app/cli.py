import argparse,json
from .config import settings
from .rag import RAG

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    i=s.add_parser("ingest"); i.add_argument("--strategy",choices=["fixed","recursive","semantic","all"],default="all"); i.add_argument("--backend",choices=["faiss","qdrant"],default=settings.vector_backend)
    a=s.add_parser("ask"); a.add_argument("question"); a.add_argument("--backend",choices=["faiss","qdrant"],default=settings.vector_backend)
    s.add_parser("voice"); b=s.add_parser("benchmark"); b.add_argument("--backend",choices=["faiss","qdrant"],default=settings.vector_backend)
    x=p.parse_args()
    if x.cmd=="ingest": print(json.dumps(RAG(x.backend).ingest(strategy=x.strategy),indent=2))
    elif x.cmd=="ask": print(json.dumps(RAG(x.backend).ask(x.question).to_dict(),indent=2,ensure_ascii=False))
    elif x.cmd=="voice":
        from .stt import record_wav,Transcriber
        from .tts import speak
        print("Recording 6 seconds — speak now...")
        path=record_wav(); text,conf=Transcriber().transcribe_file(path); print("Transcript:",text,"confidence:",round(conf,2))
        if text:
            r=RAG().ask(text,transcript=text); print(json.dumps(r.to_dict(),indent=2,ensure_ascii=False))
            if r.should_answer:speak(r.answer)
    else:
        from .benchmark import run; print(json.dumps(run(x.backend),indent=2))
if __name__=="__main__": main()

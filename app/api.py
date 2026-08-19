import os,tempfile
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .rag import RAG
from .stt import Transcriber

app=FastAPI(title="VoiceRAG")
rag=None; transcriber=None

class AskRequest(BaseModel):
    question:str

@app.on_event("startup")
def startup():
    global rag,transcriber
    rag=RAG(); transcriber=Transcriber()

@app.get("/",response_class=HTMLResponse)
def home():
    return '''<!doctype html><html><head><meta charset="utf-8"><title>VoiceRAG</title>
<style>body{font-family:Arial;max-width:800px;margin:40px auto;padding:20px}button{padding:12px 18px;margin:5px}pre{white-space:pre-wrap;background:#f5f5f5;padding:15px;border-radius:10px}</style></head>
<body><h1>🎤 VoiceRAG</h1><p>Speak a question → transcribe → retrieve → guardrail → grounded answer.</p>
<button onclick="start()">🎙 Start</button><button onclick="stop()">⏹ Stop</button><p id="status">Ready</p><pre id="out"></pre>
<script>
let rec,chunks=[];
async function start(){chunks=[];let s=await navigator.mediaDevices.getUserMedia({audio:true});rec=new MediaRecorder(s);
rec.ondataavailable=e=>chunks.push(e.data);rec.onstop=async()=>{let fd=new FormData();fd.append("audio",new Blob(chunks,{type:"audio/webm"}),"q.webm");
document.getElementById("status").textContent="Processing...";let r=await fetch("/voice",{method:"POST",body:fd});let j=await r.json();document.getElementById("out").textContent=JSON.stringify(j,null,2);document.getElementById("status").textContent="Done"};rec.start();document.getElementById("status").textContent="Recording..." }
function stop(){if(rec)rec.stop();}
</script></body></html>'''

@app.post("/ask")
def ask(req:AskRequest):
    try:return rag.ask(req.question).to_dict()
    except Exception as e:raise HTTPException(500,str(e))

@app.post("/voice")
async def voice(audio:UploadFile=File(...)):
    suffix=os.path.splitext(audio.filename or ".webm")[1] or ".webm"; fd,path=tempfile.mkstemp(suffix=suffix); os.close(fd)
    try:
        with open(path,"wb") as f:f.write(await audio.read())
        text,conf=transcriber.transcribe_file(path)
        if not text:raise HTTPException(400,"No speech detected")
        out=rag.ask(text,transcript=text).to_dict(); out["transcription_confidence"]=conf; return out
    finally:
        try:os.remove(path)
        except OSError:pass

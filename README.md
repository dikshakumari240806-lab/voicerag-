# VoiceRAG — Fast, Grounded Voice Intelligence

Complete pipeline: microphone/browser voice → Whisper transcription → 3 chunking strategies → embeddings → FAISS or local Qdrant → grounded LLM → guardrails → sources → optional TTS → benchmark harness.

## Setup
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Put your `OPENAI_API_KEY` in `.env`.

Add `.txt`, `.md`, `.pdf`, or `.docx` files to `data/`.

## Build index
```powershell
python -m app.cli ingest --strategy all --backend faiss
```
Or:
```powershell
python -m app.cli ingest --strategy all --backend qdrant
```

## Ask
```powershell
python -m app.cli ask "What is the scholarship deadline?"
python -m app.cli voice
```

## Web UI
```powershell
uvicorn app.api:app --reload
```
Open http://127.0.0.1:8000 and press Start recording.

## Benchmark
```powershell
python -m app.cli benchmark --backend faiss
```
Results are saved to `benchmarks/results.json`.

### Three chunkers
`fixed` = overlapping windows; `recursive` = paragraph/sentence-aware; `semantic` = embedding-similarity grouping. `all` indexes all three and records the strategy in metadata.

### Guardrails
- empty/invalid query rejection
- retrieval-score threshold
- prompt-injection treated as data, not instructions
- grounded-only generation
- refusal when evidence is insufficient
- source IDs returned with successful answers

### Latency
P50/P70/P100 are measured by the harness. Do not claim <200 ms unless your actual environment achieves it. Cloud generation and local Whisper can dominate end-to-end latency.

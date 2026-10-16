# RAG Lab – IT Problem Management Knowledge Base

RAG implementation using my own data: a small IT known-error knowledge base.

- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2`
- **Vector DB:** Qdrant (in-memory)
- **LLM:** Phi-2 via Llamafile (local, `localhost:8080`)

## Run
1. Start the LLM: `phi-2.llamafile.exe --server --nobrowser --port 8080 -c 2048`
2. `python rag_app.py`
3. Ask e.g. "Users keep getting kicked off the VPN"

## Example
Q: VPN → retrieved "VPN drops every 30 min" (score 0.506)
A: The likely root cause is an IKE phase-2 rekey timeout mismatch... align the phase-2 lifetime on the firewall and VPN client.
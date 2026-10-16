import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# 1. Load your data as a list of dictionaries
#    Each CSV row becomes one dict, e.g. {"title": "...", "symptom": "...", ...}
data = pd.read_csv("my_data.csv").fillna("").to_dict("records")
print(f"Loaded {len(data)} records")

# 2. Embedding model (turns text into vectors)
#    all-MiniLM-L6-v2 is small and fast; it maps any text to a 384-number vector
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Combine the useful fields of a record into one sentence to embed
def to_text(d):
    return f"{d['title']}. {d['symptom']}. Cause: {d['root_cause']}. Fix: {d['resolution']}"

# 3. Vector database (in memory) + upload records
#    COSINE distance = similarity by meaning; scores range ~0 (unrelated) to 1 (identical)
qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name="kb",
    vectors_config=models.VectorParams(
        size=encoder.get_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)
# Each point = an id + the record's vector + the original record (payload)
qdrant.upload_points(
    collection_name="kb",
    points=[
        models.PointStruct(id=i, vector=encoder.encode(to_text(d)).tolist(), payload=d)
        for i, d in enumerate(data)
    ],
)

# 4. Local LLM (Llamafile) endpoint
#    /completion is Llamafile's native text-completion API - it accepts a raw prompt,
#    which suits Phi-2 (a non-chat model) better than the chat API
LLM_URL = "http://localhost:8080/completion"

def ask(question):
    # Retrieve: embed the question and find the 3 most similar records
    hits = qdrant.query_points(
        collection_name="kb",
        query=encoder.encode(question).tolist(),
        limit=3,
    ).points

    print("\n--- Retrieved records ---")
    for h in hits:
        print(f"{h.score:.3f}  {h.payload['title']}")

    # Filter: keep only records that really match (score >= 0.3)
    #    Less text for Phi-2 = faster and more focused answers
    relevant = [h.payload for h in hits if h.score >= 0.3]
    if not relevant:
        print("\n--- LLM answer ---\nNo matching record in the knowledge base.")
        return
    context = "\n".join(f"- {to_text(d)}" for d in relevant)

    # Augment: build the prompt in Phi-2's native "Instruct: ... Output:" format
    prompt = (
        "Instruct: You are an IT problem-management assistant. Using only the knowledge "
        "base records below, state the likely root cause and the recommended fix.\n\n"
        f"Knowledge base records:\n{context}\n\n"
        f"Problem: {question}\n"
        "Output:"
    )

    # Generate: send the prompt to Llamafile
    #    n_predict=150   -> max length of the answer (shorter = faster on CPU)
    #    temperature=0.2 -> factual, not creative
    #    stop=[...]      -> stop before the model invents a new question
    #    timeout=180     -> give up after 3 minutes instead of hanging forever
    print("\n(thinking... this can take up to a minute on CPU)")
    resp = requests.post(
        LLM_URL,
        json={
            "prompt": prompt,
            "n_predict": 150,
            "temperature": 0.2,
            "stop": ["Instruct:", "\n\n\n"],
        },
        timeout=180,
    )
    resp.raise_for_status()
    print("\n--- LLM answer ---")
    print(resp.json()["content"].strip())

# 5. Ask questions in a loop (press Enter on an empty line or type 'exit' to quit)
while True:
    q = input("\nAsk a question (or 'exit'): ").strip()
    if q.lower() in ("exit", "quit", ""):
        break
    ask(q)
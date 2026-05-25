import streamlit as st

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams

from groq import Groq


# ------------------------
# CONFIGURATION
# ------------------------

model = SentenceTransformer("BAAI/bge-m3")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="TP2_RAG",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

documents = [
    "Le machine learning est une branche de l'intelligence artificielle.",

    "Python est très utilisé en data science.",

    "Qdrant est une base de données vectorielle."
]

vectors = model.encode(documents)

payload = [
    {"text": documents[0]},
    {"text": documents[1]},
    {"text": documents[2]}
]

client.upload_collection(
    collection_name="TP2_RAG",
    vectors=vectors,
    payload=payload,
    ids=[1, 2, 3]
)

client_groq = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ------------------------
# FONCTION RAG
# ------------------------

def rag(question):

    query_vector = model.encode(question)

    results = client.query_points(
        collection_name="TP2_RAG",
        query=query_vector,
        limit=2
    )

    context = ""

    for r in results.points:
        context += r.payload["text"] + "\n"

    prompt = f"""
    Contexte :
    {context}

    Question :
    {question}

    Réponds clairement en français.
    """

    chat_completion = client_groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


# ------------------------
# INTERFACE STREAMLIT
# ------------------------

st.title("TP2 - RAG")

question = st.text_input("Posez votre question")

if st.button("Envoyer"):

    response = rag(question)

    st.write(response)

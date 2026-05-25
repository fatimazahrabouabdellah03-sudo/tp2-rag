import streamlit as st

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams

from groq import Groq


# ------------------------
# CONFIGURATION
# ------------------------

model = SentenceTransformer("BAAI/bge-m3")

client = QdrantClient(
    url="TON_URL_QDRANT",
    api_key="TA_API_KEY"
)

client.create_collection(
    collection_name="TP2_RAG",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

documents = [
   # Français
        "Le machine learning est une branche de l'intelligence artificielle.",
        "Python est très utilisé en data science et en traitement du langage naturel.",
        "Qdrant est une base de données vectorielle performante et open-source.",
        "Le RAG combine la recherche dans une base de connaissances avec la génération de texte.",
        "Les embeddings permettent de représenter le sens d'un texte sous forme de vecteurs.",
        # English
        "Machine learning enables computers to learn from data without explicit programming.",
        "Vector databases store and retrieve high-dimensional embeddings efficiently.",
        "Retrieval-Augmented Generation improves LLM accuracy using external knowledge.",
        "The BAAI/bge-m3 model supports multilingual embeddings with 1024 dimensions.",
        "Groq provides fast inference for large language models like Llama 3.3 70B.",
        # العربية
        "التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يتعلم من البيانات.",
        "قواعد البيانات المتجهة تُستخدم لتخزين التضمينات واسترجاعها بسرعة.",
        "يجمع نظام RAG بين البحث في قاعدة المعرفة وتوليد النصوص.",
        "نموذج bge-m3 يدعم اللغة العربية والفرنسية والإنجليزية.",
        "Groq يوفر استدلالاً سريعاً لنماذج اللغة الكبيرة.",
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

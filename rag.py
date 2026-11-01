import os
import pickle
from pathlib import Path
import fitz
# import pymupdf
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
load_dotenv()
# BASE DIRECTORIES
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data"
RAG_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# RAG STORE FILES
GENERAL_STORE_FILE = (
    RAG_DIR / "general_rag_store.pkl"
)
TECHNICAL_STORE_FILE = (
    RAG_DIR / "technical_rag_store.pkl"
)
# MODELS
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"
# RAG SETTINGS
TOP_K = 4
RELEVANCE_THRESHOLD = 0.20
QA_MIN_SCORE = 0.20

OUT_OF_DOMAIN_MESSAGE = (
    "Sorry, I could not find information relevant to your question in the documentation."
)
NO_EXACT_ANSWER_MESSAGE = (
    "I could not find an exact answer to this question "
    "in the provided dental documentation."
)
UNAUTHORIZED_MESSAGE = (
    "Unauthorized: You do not have permission to access "
    "this knowledge base."
)
# MODEL CACHE
_embedding_model = None
_qa_model = None
# EMBEDDING MODEL
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(
            "Loading embedding model..."
        )
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )
        print(
            "Embedding model loaded."
        )
    return _embedding_model
# EXTRACTIVE QA MODEL
def get_qa_model():
    global _qa_model
    if _qa_model is None:
        print(
            "Loading QA model..."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            QA_MODEL
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            QA_MODEL
        )
        model.eval()
        _qa_model = {
            "tokenizer": tokenizer,
            "model": model
        }
        print(
            "Extractive QA model loaded."
        )
    return _qa_model
# CHUNK TEXT
def split_text(
    text,
    chunk_size=200,
    overlap=30
):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(
            words[start:end]
        )
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
# GET RAG STORE
def get_store_file(doctype):
    if doctype == "general":
        return GENERAL_STORE_FILE
    if doctype == "technical":
        return TECHNICAL_STORE_FILE
    raise ValueError(
        "Invalid document type. "
        "Use 'general' or 'technical'."
    )
# CREATE RAG INDEX
def create_index(
    text,
    doctype
):
    chunks = split_text(text)
    if not chunks:
        raise ValueError(
            "Document does not contain usable text."
        )
    model = get_embedding_model()
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    store = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    store_file = get_store_file(
        doctype
    )
    with open(
        store_file,
        "wb"
    ) as file:
        pickle.dump(
            store,
            file
        )
    print(
        f"Created {len(chunks)} chunks for {doctype}"
    )
    return len(chunks)

# DOCUMENT DIRECTORY
DOCUMENT_DIR = BASE_DIR / "documents"
GENERAL_DOCUMENT_DIR = (
    DOCUMENT_DIR / "general"
)
TECHNICAL_DOCUMENT_DIR = (
    DOCUMENT_DIR / "technical"
)
GENERAL_DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
TECHNICAL_DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# LOAD BACKEND PDF
def load_backend_document(
    doctype
):
    # Select document directory
    if doctype == "general":
        document_dir = GENERAL_DOCUMENT_DIR
    elif doctype == "technical":
        document_dir = TECHNICAL_DOCUMENT_DIR
    else:
        raise ValueError(
            "Invalid document type. "
            "Use 'general' or 'technical'."
        )
    # Find PDFs
    pdf_files = list(
        document_dir.glob("*.pdf")
    )
    if not pdf_files:
        raise FileNotFoundError(
            f"No {doctype} PDF document was found "
            "in the document folder."
        )
    # Only one PDF should exist
    if len(pdf_files) > 1:
        raise ValueError(
            f"Only one {doctype} PDF document should "
            "be present in the document folder."
        )
    file_path = pdf_files[0]
    print(
        f"Reading {doctype} document:",
        file_path
    )
    # Open PDF
    document = fitz.open(
        file_path
    )
    content = ""
    # Extract text page by page
    for page in document:
        content += page.get_text()
    document.close()
    # Validate content
    content = content.strip()
    if not content:
        raise ValueError(
            f"{file_path.name} does not contain "
            "readable text.")
    # Chunk + embedding + pickle
    chunk_count = create_index(
        content,
        doctype
    )
    print(
        f"Successfully indexed {doctype} "
        f"{file_path.name}"
    )
    return chunk_count
# LOAD RAG INDEX
def load_index(
    doctype
):
    store_file = get_store_file(
        doctype
    )
    if not store_file.exists():
        return None
    with open(
        store_file,
        "rb"
    ) as file:
        store = pickle.load(file)
    return store
# RETRIEVE RELEVANT CHUNKS
def retrieve_chunks(
    question,
    doctype,
    top_k=TOP_K
):
    store = load_index(
        doctype
    )
    if store is None:
        return []
    chunks = store["chunks"]
    embeddings = store["embeddings"]
    model = get_embedding_model()
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    # Cosine similarity
    # Because embeddings are normalized,
    # dot product = cosine similarity
    scores = np.dot(
        embeddings,
        question_embedding
    )
    # Highest similarity first
    top_indices = np.argsort(
        scores
    )[::-1][:top_k]
    results = []
    for index in top_indices:
        results.append(
            {
                "text": chunks[index],
                "score": float(
                    scores[index]
                )
            }
        )
    return results
# CHECK QUESTION RELEVANCE
def is_relevant_question(
    retrieved_chunks
):
    if not retrieved_chunks:
        return False
    best_score = max(
        item["score"]
        for item in retrieved_chunks
    )
    print(
        "Best retrieval similarity:",
        best_score
    )
    return (
        best_score >= RELEVANCE_THRESHOLD
    )
# EXTRACT EXACT ANSWER
def extract_exact_answer(
    question,
    retrieved_chunks
):
    if not retrieved_chunks:
        return None
    qa = get_qa_model()
    tokenizer = qa["tokenizer"]
    model = qa["model"]
    best_answer = None
    best_score = 0.0
    # Process each retrieved chunk
    for item in retrieved_chunks:
        context = item["text"]
        try:
            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=500,
                return_offsets_mapping=True)
            offset_mapping = inputs.pop(
                "offset_mapping"
            )
            sequence_ids = inputs.sequence_ids(
                0
            )
            with torch.no_grad():
                outputs = model(**inputs)
            start_logits = (
                outputs.start_logits[0]
            )
            end_logits = (
                outputs.end_logits[0]
            )
            # Only tokens belonging to context
            context_indexes = [
                i
                for i, sequence_id in enumerate(
                    sequence_ids
                )
                if sequence_id == 1
            ]
            if not context_indexes:
                continue
            # Find best start/end answer
            for start_index in context_indexes:
                for end_index in context_indexes:
                    # End cannot come before start
                    if end_index < start_index:
                        continue
                    # Maximum answer length
                    if (
                        end_index
                        - start_index
                        + 1
                        > 25
                    ):
                        continue
                    start_offset = (
                        offset_mapping[
                            0
                        ][start_index]
                    )
                    end_offset = (
                        offset_mapping[
                            0
                        ][end_index]
                    )
                    start_char = (
                        start_offset[0].item()
                    )
                    end_char = (
                        end_offset[1].item()
                    )
                    if end_char <= start_char:
                        continue
                    answer = context[
                        start_char:end_char
                    ].strip()
                    if not answer:

                        continue
                    # Calculate QA score
                    score = (
                        start_logits[
                            start_index
                        ].item()
                        +
                        end_logits[
                            end_index
                        ].item()
                    )
                    if score > best_score:
                        best_score = score
                        best_answer = answer
        except Exception as e:
            print(
                "Extractive QA error:",
                str(e)
            )
            continue
    # Debug output
    print(
        "Best Extractive QA score:",
        best_score
    )
    print(
        "Best Extractive QA answer:",
        best_answer
    )
    # No answer
    if best_answer is None:
        return None
    # Low confidence
    if best_score < QA_MIN_SCORE:
        print(
            "QA confidence too low."
        )
        return None
    return best_answer
# GENERATE ANSWER
def generate_answer(
    question,
    retrieved_chunks
):
    # No retrieved chunks
    if not retrieved_chunks:
        return OUT_OF_DOMAIN_MESSAGE
    # Relevance check
    if not is_relevant_question(
        retrieved_chunks
    ):
        return OUT_OF_DOMAIN_MESSAGE
    # First attempt:
    # Extractive QA
    exact_answer = extract_exact_answer(
        question,
        retrieved_chunks
    )
    if exact_answer:
        return exact_answer
    # Build context for LLM fallback
    context = "\n\n".join(
        [
            (
                f"DOCUMENT SECTION {i + 1}:\n"
                f"{item['text']}"
            )
            for i, item in enumerate(
                retrieved_chunks
            )
        ]
    )
    # OpenRouter API key
    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )
    print(
        "OpenRouter key loaded:",
        bool(api_key)
    )
    if not api_key:
        return NO_EXACT_ANSWER_MESSAGE
    # OpenRouter client
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    # Prompt
    prompt = f"""


IMPORTANT RULES:

You are the Technical and Product Support Assistant for MyLabConnect.
Your goal is to provide clear, accurate, and actionable solutions to the user's question using ONLY the retrieved document context below.

CONTEXT POLICIES & CONSTRAINTS:
1. STRICT ADHERENCE: Rely ONLY on the clear facts directly mentioned in the DOCUMENT CONTEXT. Do not introduce outside technical knowledge, assumptions, or guesswork.
2. TROUBLESHOOTING & PROCEDURES: If the user is asking about a technical issue, error code (e.g., HTTP 500, login failure, connection timeouts), or status, outline the exact step-by-step diagnostic/resolution steps provided in the context.
3. CONCISE & READABLE: Be direct and clear. Use bullet points or numbered lists if the answer involves multiple steps or conditions.
4. OUT-OF-CONTEXT FALLBACK: If the provided context does NOT contain enough information to answer the question, or if the question is completely unmentioned, respond EXACTLY with:
"I could not find this information in the provided documentation."
5. FORMATTING RESTRICTIONS: Do not refer to "DOCUMENT SECTION", "Context", or "the provided text". Present the response naturally as a direct answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Return ONLY the direct answer.
"""
    # Call OpenRouter
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user","content": prompt}])
        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )
        if not answer:
            return NO_EXACT_ANSWER_MESSAGE
        return answer
    except Exception as e:
        print(
            "OpenRouter error occurred:",
            str(e)
        )
        return NO_EXACT_ANSWER_MESSAGE
# MAIN RAG FUNCTION
def ask_rag(
    question,
    doctype,
    user_role=None
):
    # Validate document type
    if doctype not in [
        "general",
        "technical"
    ]:
        return {
            "answer": (
                "Invalid document type."
            ),
            "sources": []
        }
    # Authorization
    # admin can access both
    # support can access support only
    # product can access product only
    if user_role:
        if user_role == "superadmin":
            pass

        elif user_role== "support_admin":
            if doctype != "technical":
                return {
                    "answer": UNAUTHORIZED_MESSAGE,
                    "sources": []
                }

        elif user_role == "product_admin":
            if doctype != "general":
                return{
                    "answer": UNAUTHORIZED_MESSAGE,
                    "sources": []
                }
        elif user_role == "general_user":
            if doctype != "general":
                return {
                    "answer": UNAUTHORIZED_MESSAGE,
                    "sources": []
                }
        else:
            return {
                "answer": UNAUTHORIZED_MESSAGE,
                "sources": []
            }

        retrieved_chunks = retrieve_chunks(
            question,
            doctype,
            top_k=TOP_K
        )

        # Reject unrelated questions
        if not is_relevant_question(
                retrieved_chunks
        ):
            return {
                "answer": OUT_OF_DOMAIN_MESSAGE,
                "sources": []
            }
    # Generate answer
        answer = generate_answer(
            question,
            retrieved_chunks
        )
    # Return result
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
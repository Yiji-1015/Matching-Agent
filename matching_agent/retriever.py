from pathlib import Path
from typing import Iterable

import faiss
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTORSTORE_DIR


def make_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


def load_retriever(
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    k: int = 5,
):
    embeddings = make_embeddings(embedding_model)
    db = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(1536),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    db = db.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_kwargs={"k": k})


def get_candidates(query: str, username: str, retriever) -> list[str]:
    if hasattr(query, "content"):
        query = query.content

    results = retriever.invoke(query)
    return [
        f"{doc.metadata['User']}: {doc.page_content}"
        for doc in results
        if doc.metadata.get("User") != username
    ]


def build_vectorstore_from_excel(
    dataset_path: Path,
    output_dir: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    message_col: str = "Message",
    user_col: str = "User",
) -> None:
    """Rebuild the FAISS vectorstore from the source Excel dataset."""
    df = pd.read_excel(dataset_path)
    embeddings = make_embeddings(embedding_model)
    db = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(1536),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    for i, row in df.iterrows():
        db.add_texts(
            [row[message_col]],
            metadatas=[{"User": row[user_col]}],
            ids=[str(i)],
        )

    db.save_local(str(output_dir))


def dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))

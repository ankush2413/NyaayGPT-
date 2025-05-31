import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from uuid import uuid4
from tqdm import tqdm
persist_dir = "vectorstore/chroma_store"

def save_uploaded_file(file, session_id):
    folder = f"embed_data/uploads/{session_id}"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path

persist_base_dir = "vectorstore/chroma_store"

def process_and_embed(file_path, session_id):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = text_splitter.split_documents(pages)

    # Prepare Chroma DB
    persist_dir = f"{persist_base_dir}/{session_id}"
    os.makedirs(persist_dir, exist_ok=True)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        embedding_function=embeddings_model,
        persist_directory=persist_dir
    )

    # Batch embed and store
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(all_docs), BATCH_SIZE), desc="Embedding"):
        batch = all_docs[i:i + BATCH_SIZE]
        ids = [str(uuid4()) for _ in batch]
        vectorstore.add_documents(documents=batch, ids=ids)

    vectorstore.persist()
    return len(all_docs)


def process_and_embed_without_batch(file_path, session_id):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"{persist_dir}/{session_id}"
    )
    vectordb.persist()
    return len(docs)

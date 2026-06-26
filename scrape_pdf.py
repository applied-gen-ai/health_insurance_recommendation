import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from langchain.schema import Document
import requests
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
import pdfplumber
from io import BytesIO


embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def read_pdf(pdf_path):
    # Variable to store the complete text

    response = requests.get(pdf_path)
    response.raise_for_status()  # Raise an error if download fails

    # Step 2: Load the PDF into pdfplumber
    pdf_file = BytesIO(response.content)

    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    # Step 3: Print or Process the Extracted Text
    return text
    

def insert_text(text, provider, policy_name):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    )

    chunks = text_splitter.create_documents([text])
    print(len(chunks), type(chunks))
    documents = []

    for chun in chunks:
        metadata = {"provider": provider, "policy_name": policy_name}
        doc = Document(page_content=chun.page_content, metadata=metadata)
        documents.append(doc)

    dimension = 384  # Default dimension for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.save_local("faiss_index")
    print("inserted...........")




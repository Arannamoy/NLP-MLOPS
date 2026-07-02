# rag_assistant.py
# Compatible with: langchain>=0.3, langchain-community>=0.3, langchain-ollama, faiss-cpu
# LLM: Ollama (phi4:latest)

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

import os

# -------------------------------
# 🧩 Step 1: Load Documents
# -------------------------------
# Example text file — replace this with your own path
DOC_PATH = "docs.txt"

if not os.path.exists(DOC_PATH):
    with open(DOC_PATH, "w") as f:
        f.write("This is a sample document. Replace it with your own data for retrieval-augmented generation (RAG).")

loader = TextLoader(DOC_PATH)
documents = loader.load()

# -------------------------------
# ✂️ Step 2: Split into Chunks
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(documents)

# -------------------------------
# 🔍 Step 3: Create Vectorstore
# -------------------------------
embeddings = OllamaEmbeddings(model="phi3")  # or "mxbai-embed-large" if installed in Ollama
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 🧠 Step 4: Initialize LLM
# -------------------------------
llm = OllamaLLM(model="phi4:latest")

# -------------------------------
# 🤖 Step 5: Build RAG Chain
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# 💬 Step 6: Ask Questions
# -------------------------------
while True:
    query = input("\n🧠 Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        print("Goodbye 👋")
        break

    result = qa_chain.invoke({"query": query})
    print("\n💡 Answer:\n", result["result"])

    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "unknown"))

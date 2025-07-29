import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import tempfile

# Load document using LangChain loaders
def load_document(uploaded_file):
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path=uploaded_file.name)
    else:
        loader = UnstructuredWordDocumentLoader(file_path=uploaded_file.name)
    return loader.load()

# Set up RAG chain with Chroma vector store (no external install needed)
def setup_rag_chain(docs, api_key):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return rag_chain

# Streamlit app
def main():
    st.title("💬 Chat with Your Document")

    api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

    if api_key:
        uploaded_file = st.file_uploader("📄 Upload a PDF or Word document", type=["pdf", "docx"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            uploaded_file.name = tmp_file_path  # Patch for LangChain loaders
            docs = load_document(uploaded_file)
            rag_chain = setup_rag_chain(docs, api_key)

            user_input = st.text_input("💬 Ask something about your document")

            if user_input:
                response = rag_chain.run(user_input)
                st.write("🧠", response)

if __name__ == "__main__":
    main()

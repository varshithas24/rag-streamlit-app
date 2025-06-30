import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import hashlib

# 🧠 Chroma settings for Streamlit Cloud (important fix)
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("📘 PDF Q&A - Gemini RAG with Chroma")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    with st.spinner("🔍 Preparing data..."):
        # Create unique hash for uploaded file
        file_bytes = pdf_file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        persist_dir = f"db_persist/vectorstore_{file_hash}"

        os.makedirs("temp_pdf", exist_ok=True)
        temp_path = os.path.join("temp_pdf", "uploaded.pdf")

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Use embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # If vector DB already exists
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            st.info("🧠 Reusing saved vectorstore...")
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                client_settings=Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
            )
        else:
            st.info("📚 Loading and chunking PDF...")
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)

            st.info("💾 Creating vectorstore...")
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
                client_settings=Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
            )
            vectorstore.persist()
            st.success("✅ Vectorstore created and saved!")

        # Create retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        # Prompt template
        system_prompt = (
            "You are an expert assistant for answering questions using the provided context. "
            "Use the following context to give a thorough and detailed answer to the user's query. "
            "Include examples or code snippets when appropriate.\n\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # LLM + RAG pipeline
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Ask question
        question = st.text_input("❓ Ask a question about the PDF", placeholder="Type your question here...")
        if question:
            with st.spinner("💬 Generating answer..."):
                response = rag_chain.invoke({"input": question})
                st.subheader("🧠 Answer:")
                st.success(response.get("answer", "⚠️ No answer found."))

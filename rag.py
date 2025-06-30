import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.title("📘 PDF Q&A – Gemini RAG (FAISS)")

pdf_file = st.file_uploader("Upload a PDF (must contain selectable text)", type="pdf")

if pdf_file:
    with st.spinner("🔍 Extracting text…"):
        # Save temp file
        os.makedirs("temp_pdf", exist_ok=True)
        temp_path = os.path.join("temp_pdf", "uploaded.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())

        # ---- 1️⃣  Try PyPDFLoader, fall back to PDFMiner
        try:
            pages = PyPDFLoader(temp_path).load()
        except Exception:
            pages = []

        if not pages:  # fallback if no pages or empty
            try:
                pages = PDFMinerLoader(temp_path).load()
            except Exception:
                pages = []

        if not pages:
            st.error("❌ Couldn’t extract any text from this PDF. "
                     "Make sure it isn’t just scanned images.")
            st.stop()

        # ---- 2️⃣  Chunk + embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        if not docs:
            st.error("❌ No extractable text chunks found in the PDF.")
            st.stop()

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        # Prompt & chain
        system_prompt = (
            "You are an expert assistant. Use the provided context to answer the user’s question.\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

    # ---- 3️⃣  Q&A UI
    question = st.text_input("❓ Ask a question about the PDF")
    if question:
        with st.spinner("💬 Generating answer…"):
            answer = rag_chain.invoke({"input": question}).get("answer", "⚠️ No answer found.")
        st.subheader("🧠 Answer")
        st.success(answer)

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

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("📘 PDF Q&A - Gemini RAG")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    with st.spinner("🔍 Preparing data..."):
        # Save temp PDF file
        os.makedirs("temp_pdf", exist_ok=True)
        temp_path = os.path.join("temp_pdf", "uploaded.pdf")

        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())

        # Load and split PDF
        st.info("📚 Loading and chunking PDF...")
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Create in-memory vectorstore
        st.info("💾 Creating in-memory vectorstore...")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

        # Retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        # Prompt
        system_prompt = (
            "You are an expert assistant for answering questions using the provided context. "
            "Use the following context to give a thorough and detailed answer to the user's query. "
            "Include examples or code snippets when appropriate.\n\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # LLM and chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Ask a question
        question = st.text_input("❓ Ask a question about the PDF", placeholder="Type your question here...")
        if question:
            with st.spinner("💬 Generating answer..."):
                response = rag_chain.invoke({"input": question})
                st.subheader("🧠 Answer:")
                st.success(response.get("answer", "⚠️ No answer found."))

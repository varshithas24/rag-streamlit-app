# 📘 PDF Q&A with RAG (Gemini + FAISS + Streamlit)

This is a Streamlit-based web application that allows users to upload PDF files and ask questions based on their content using **Retrieval-Augmented Generation (RAG)** with **Google's Gemini 2.0 Flash model** and **FAISS** vector storage.

---

## 🚀 Features

- ✅ **PDF Upload**: Upload any text-based PDF file.
- 🔍 **Text Chunking**: Splits the PDF into context chunks for better semantic search.
- 🧠 **Embeddings**: Uses Google's `text-embedding-004` model for vector representation.
- 🔎 **FAISS Vector Store**: In-memory vector database optimized for Streamlit Cloud deployment.
- 💬 **Gemini 2.0 Flash**: Generates context-aware answers based on retrieved PDF content.
- ⚡ **Fast Response**: Optimized for quick Q&A interactions.

---

## 🧠 How It Works

1. **PDF Upload**: Users upload a text-based PDF.
2. **Text Extraction**: The PDF is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding**: Each chunk is embedded using Google’s `text-embedding-004`.
4. **Retrieval**: Relevant chunks are retrieved using **FAISS** based on the user’s question.
5. **Generation**: The chunks + question are passed to **Gemini 2.0 Flash** to generate the answer.

---

## 🛠️ Tech Stack

- `Streamlit` – Frontend for interactive UI
- `LangChain` – RAG pipeline and document management
- `GoogleGenerativeAI` – Gemini 2.0 Flash + Embedding model
- `FAISS` – Vector similarity search engine
- `PyPDFLoader` – Extracts text from PDF files
- `dotenv` – Manages API keys securely

---

## 📂 Project Structure

```
├── rag.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env                 # Contains your Google API key (not committed)
├── temp_pdf/            # Temp folder for uploaded PDFs
```

---

## ✅ Prerequisites

- Python 3.8 or later
- Google API Key for [Generative AI](https://aistudio.google.com/app/apikey)

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pdf-qna-gemini.git
   cd pdf-qna-gemini
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key to a `.env` file**
   ```
   GOOGLE_API_KEY="your_real_api_key_here"
   ```

---

## ▶️ Run the App Locally

```bash
streamlit run rag.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and select your repo.
4. Set your `GOOGLE_API_KEY` in **Secrets** (Settings > Secrets):
   ```
   GOOGLE_API_KEY = "your_real_api_key_here"
   ```

---

## 🧪 Example Use Case

- Upload a PDF like a research paper or study notes.
- Ask questions like:
  - "What is the main topic?"
  - "List the key findings"
  - "What are the symptoms of diabetes?"
- Get direct, context-aware answers in seconds.

---

## 📈 Future Improvements

- Persistent vector DB for session reuse
- OCR support for scanned PDFs
- Support for multiple PDFs
- Multi-language predictions
- Document summarization and export options

---

## 📄 Web Link

https://rag-app-gemini-sv.streamlit.app/
---

## 🤝 Acknowledgments

- [Google Generative AI](https://aistudio.google.com/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

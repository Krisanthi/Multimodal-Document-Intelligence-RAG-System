
# DocIntel RAG: Multimodal Document Intelligence System

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![AWS](https://img.shields.io/badge/AWS%20S3%20%26%20Lambda-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
  ![OpenSearch](https://img.shields.io/badge/OpenSearch-005EB8?style=for-the-badge&logo=opensearch&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
  ![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)
  ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

DocIntel RAG is a production-ready Retrieval-Augmented Generation (RAG) system designed for document Q&A. The application processes PDFs, images, and scanned documents, allowing users to ask questions in plain English and receive accurate, citation-backed answers powered by AI.

**Live Public Demo:** [Access the deployed Streamlit Cloud App here](https://multimodal-document-intelligence-rag-system-j9xgrgx8ejsucrmhwx.streamlit.app/#doc-intel-rag). This hosted version is publicly accessible without requiring any local installation.

## Technical Highlights

* **Dual Vector Store Architecture:** Features a dynamically configurable backend to support different deployment environments. **FAISS** is utilized for the live, zero-infrastructure Streamlit Cloud deployment, while **OpenSearch** is provided for production-grade hybrid search (BM25 + KNN) via local Docker environments.
* **Advanced RAG Pipeline:** Implements a complete ingestion and querying pipeline utilizing sentence-transformers for text embeddings and a cross-encoder model for fine-grained results reranking.
* **AWS Cloud Integration:** Utilizes AWS S3 for secure, temporary staging of uploaded documents before automatic deletion, with optional AWS SAM deployment capabilities for Lambda and API Gateway.
* **Built-in Evaluation Framework:** Features an automated evaluation suite to measure retrieval quality (Precision@K, Recall@K, MRR, F1) and answer quality (Semantic Similarity, Keyword Accuracy).

## System Architecture

The application handles heavy data processing and inference through two distinct workflows.

**1. Ingestion Flow (Document Upload)**
* Documents are uploaded and staged temporarily in an AWS S3 bucket.
* Local parsers (pdfplumber and pytesseract) extract text, tables, and layouts at zero API cost.
* Extracted text is split into overlapping 512-character chunks to maintain context.
* Chunks are converted into 384-dimensional vectors using the `all-MiniLM-L6-v2` model and loaded into the active vector database (FAISS for cloud, OpenSearch for local).

**2. Query Flow (Information Retrieval)**
* User questions are embedded and queried against the vector store using keyword and vector similarity techniques.
* The `ms-marco-MiniLM-L-6-v2` reranker re-scores the retrieved chunks for optimal relevance.
* The top 5 chunks are passed to the Groq Llama 3.3 70B LLM to generate a factual paragraph answer strictly grounded in the provided context.

## Tech Stack

**Machine Learning & AI**
* **LLM Provider:** Groq API (Llama 3.3 70B)
* **Embeddings & Reranking:** HuggingFace (all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2)
* **Document Parsing:** pdfplumber (PDFs), pytesseract + Pillow (Image OCR)

**Backend & Cloud**
* **Cloud Infrastructure:** AWS S3, AWS Lambda, API Gateway
* **Databases:** OpenSearch 2.11, FAISS
* **Containerization:** Docker, Docker Compose

**Frontend**
* **Framework:** Streamlit

## Local Setup & Installation

If you prefer to run the full OpenSearch architecture locally instead of using the public web app, follow these steps:

**1. Clone and Configure**
```bash
git clone [https://github.com/yourusername/Multimodal-Document-Intelligence-RAG-System.git](https://github.com/yourusername/Multimodal-Document-Intelligence-RAG-System.git)
cd Multimodal-Document-Intelligence-RAG-System
cp .env.example .env
```
Update the `.env` file with your specific `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `GROQ_API_KEY`.

**2. Run via Docker (Recommended for OpenSearch)**
Ensure Docker Desktop is running, then execute:
```bash
docker-compose up --build
```
Navigate to `http://localhost:8501` to access the Streamlit interface.

**3. Run Locally (FAISS Backend)**
Ensure system dependencies like Tesseract OCR are installed, then execute:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```
Ensure `VECTOR_BACKEND="faiss"` is set in your environment variables for local execution without Docker.



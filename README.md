# Chat with Multiple PDFs using Gemini & LangChain

An AI-powered document question-answering application that allows users to upload multiple PDF files and ask questions about their content. The application uses **Google Gemini**, **LangChain**, **FAISS**, and **Streamlit** to extract document text, create vector embeddings, retrieve relevant information, and generate context-aware answers.

## Features

- Upload and process multiple PDF documents
- Extract text from PDFs using PyPDF2
- Split large documents into manageable text chunks
- Generate semantic embeddings using Google Generative AI
- Store and retrieve document embeddings using FAISS
- Ask natural-language questions about uploaded documents
- Generate context-aware responses using Google Gemini
- Interactive web interface built with Streamlit
- Evaluate RAG responses using DeepEval metrics

## Tech Stack

- **Python**
- **Streamlit** – Interactive web application interface
- **Google Gemini** – LLM for document-based question answering
- **LangChain** – RAG pipeline and prompt orchestration
- **FAISS** – Vector storage and similarity search
- **PyPDF2** – PDF text extraction
- **Google Generative AI Embeddings** – Semantic document embeddings
- **DeepEval** – Evaluation of RAG response quality

## How It Works

1. The user uploads one or more PDF documents.
2. Text is extracted from each PDF using PyPDF2.
3. The extracted text is divided into overlapping chunks using LangChain.
4. Google Generative AI embeddings are generated for the text chunks.
5. The embeddings are stored locally in a FAISS vector database.
6. When the user asks a question, FAISS performs a similarity search to retrieve relevant document chunks.
7. The relevant context and question are passed to Gemini through LangChain.
8. Gemini generates an answer based on the retrieved document context.

### RAG Workflow

```text
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Gemini Embeddings
      ↓
FAISS Vector Store
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
Gemini + LangChain
      ↓
Generated Answer
```

## Project Structure

```text
chat-with-multiple-pdfs-gemini/
│
├── app.py              # Main Streamlit application
├── evaluate.py         # DeepEval evaluation script
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment configuration
├── .gitignore          # Files excluded from Git tracking
└── README.md            # Project documentation
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/PramodSreejaManthri910/chat-with-multiple-pdfs-gemini.git
cd chat-with-multiple-pdfs-gemini
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API key

Create a `.env` file in the project directory:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Replace `your_google_api_key_here` with your Google Gemini API key.

> Never commit your actual API key to GitHub.

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload one or more PDF files from the sidebar, click **Submit & Process**, and then ask questions about the uploaded documents.

## RAG Evaluation

The project also includes an evaluation script using **DeepEval** to assess the quality of the retrieval and generated responses.

The following metrics are evaluated:

- Contextual Precision
- Contextual Recall
- Contextual Relevancy
- Answer Relevancy
- Faithfulness

Run the evaluation with:

```bash
python evaluate.py
```

These metrics help assess whether the retrieved document context is relevant and whether the generated answers remain grounded in that context.

## Future Improvements

- Support additional document formats
- Add conversational chat history
- Improve vector-store persistence and session management
- Add source citations to generated answers
- Deploy the application for public access
- Expand the RAG evaluation dataset

## Author

**Pramod Sreeja Manthri**

GitHub: [PramodSreejaManthri910](https://github.com/PramodSreejaManthri910)

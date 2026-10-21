import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDF documents."""
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


def get_text_chunks(text):
    """Split extracted text into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Create FAISS vector embeddings and save them locally."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Create the Gemini question-answering chain."""

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details.

    If the answer is not available in the provided context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )


def user_input(user_question):
    """Retrieve relevant document chunks and generate an answer."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📄"
    )

    st.header("Chat with Multiple PDFs using Gemini")

    user_question = st.text_input(
        "Ask a question from the PDF files"
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload your PDF files and click Submit & Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed successfully.")


if __name__ == "__main__":
    main()

import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDF documents."""
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def get_text_chunks(text):
    """Split extracted PDF text into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )

    return text_splitter.split_text(text)


def get_embeddings():
    """Create the Google Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


def get_vector_store(text_chunks):
    """Create and save a FAISS vector store."""
    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings,
    )

    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Create the Gemini question-answering chain."""

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the provided context.

        Provide a detailed and accurate answer based on the context.
        If the answer is not available in the context, say:
        "Answer is not available in the provided documents."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )

    return prompt | model


def user_input(user_question):
    """Retrieve relevant chunks and generate an answer."""

    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDF files first.")
        return

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docs = vector_store.similarity_search(
        user_question,
        k=4,
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = get_conversational_chain()

    response = chain.invoke(
        {
            "context": context,
            "question": user_question,
        }
    )

    st.write("Reply:", response.content)


def main():
    """Run the Streamlit application."""

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon=" ",
    )

    st.title(" Chat with Multiple PDFs")
    st.caption("Powered by Google Gemini, LangChain, and FAISS")

    if not GOOGLE_API_KEY:
        st.error(
            "Google API key not found. "
            "Add GOOGLE_API_KEY to your .env file."
        )
        st.stop()

    user_question = st.text_input(
        "Ask a question about your PDF documents"
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("Upload Documents")

        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")

            else:
                with st.spinner("Processing PDF documents..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error(
                            "No readable text was found in the uploaded PDFs."
                        )
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)

                        st.success(
                            "Documents processed successfully. "
                            "You can now ask questions!"
                        )


if __name__ == "__main__":
    main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

# Streamlit App Header
st.header("My First Chatbot")

# Sidebar for File Upload
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Process PDF File
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""

    # Extract text from PDF
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User Question Input
    user_question = st.text_input("Type your question here")

    # Perform Similarity Search & Generate Response
    if user_question:
        matched_docs = vector_store.similarity_search(user_question)

        if matched_docs:
            context = "\n".join([doc.page_content for doc in matched_docs])

            # Initialize Chat Model
            chat_model = ChatOllama(model="llama3.2")

            # Generate Response
            response = chat_model.predict(f"Answer the following question using this context:\n\n{context}\n\nQuestion: {user_question}")

            # Display Answer
            st.write("### Answer:")
            st.write(response)
        else:
            st.write("No relevant information found in the document.")

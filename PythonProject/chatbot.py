import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (first time only)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Streamlit App Header
st.header("Hello Buddy, How Can I Help You.")

# Sidebar for File Upload
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Function to detect sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] > 0:
        return "😊 Positive"
    elif score['compound'] < 0:
        return "😟 Negative"
    else:
        return "😐 Neutral"

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Use chat-style input (auto clears after sending)
user_question = st.chat_input("Type your question here...")

# Process question
if user_question:
    sentiment = get_sentiment(user_question)

    if file is not None:
        # PDF processing
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_texts(chunks, embeddings)

        matched_docs = vector_store.similarity_search(user_question)

        if matched_docs:
            context = "\n".join([doc.page_content for doc in matched_docs])
            chat_model = ChatOllama(model="llama3.2")
            response = chat_model.predict(
                f"Answer the following question using this context:\n\n{context}\n\nQuestion: {user_question}"
            )
        else:
            response = "No relevant information found in the document."
    else:
        # General chatbot mode
        chat_model = ChatOllama(model="llama3.2")
        response = chat_model.predict(user_question)

    # Save to history
    st.session_state.chat_history.append(
        {"question": user_question, "sentiment": sentiment, "answer": response}
    )

# Display full chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}  \n*Sentiment:* {chat['sentiment']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        st.write("---")

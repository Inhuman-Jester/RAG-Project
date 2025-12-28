import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from source.retrieval_and_generation_pipeline import rag_chain  # Import the chain you built for retrieval
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "testprojectv2"
index = pc.Index(index_name)

# Streamlit UI
st.title("Interactive RAG-powered Q&A")# Title of the Application
st.write("Ask a question based on the indexed knowledge base.")

# User Input
user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):# If this button is clicked, then the below code is run
    if user_query.strip():# strip is method used to remove leading, trailing spaces
        # Run the RAG pipeline
        response = rag_chain.invoke(user_query)
        
        # Display the result
        st.subheader("Answer:")
        st.write(response)
    else:# If the user clicked on the button, without any question, then the below code is run
        st.warning("Please enter a valid question.")
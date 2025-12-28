import os
from dotenv import load_dotenv
from langsmith import utils
import time
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from pinecone import Pinecone, ServerlessSpec
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts  import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(dotenv_path=".env")

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")

# The Index we are going to interact with
index_name = "testprojectv2"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Defining the Embedding model to be used for embedding the user query
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = 384 

# Enabling tracing in LangSmith
utils.tracing_is_enabled()

# Creating a method to embed the user query, and retrieve relevant past Q&A pairs from `conv_vector`
def retrieve_chat_history(input):
    logging.info("RAG stage 1: Retrieving relevant Q&A pairs from testprojectv2/conv_vector...for follow-up questions")
    question = input['question']
    user_specific_namespace = input['user_specific_namespace']
    expanded_query = input['expanded_user_query']

    if not user_specific_namespace:
        return {"question": question, "chat_history_context": ""}
    
    # logging.info("Retrieving relevant Q&A pairs from testprojectv1/conv_vector...")
    embeddedQuestion = st_model.encode([expanded_query], convert_to_numpy=True)[0].tolist()
    chat_history_results = index.query(vector=embeddedQuestion, top_k=2, namespace=user_specific_namespace, include_metadata=True)

    # only selecting the Q&A pairs with similarity threshold above 0.40
    if len(chat_history_results['matches']):
        selected_chat_history_results = [doc for doc in chat_history_results['matches'] if doc.get('score',0) > 0.4]
    else:
        selected_chat_history_results = []

    # formatting the question and retrieved Q&A pairs into a dictionary
    # we are using a dictionary with matches key, because this is the format usually returned by a retriever, and this is format the formatContext method expects
    formatted_chat_history_results = formatContext({'matches': selected_chat_history_results})
    return {"question": question, "chat_history_context": formatted_chat_history_results}


# Creating a method to embed the (user query combined with the chat history context), and retrieve the relevant documents from `ns1`
def retrieve_relevant_docs(input):
    logging.info("RAG stage 2: Retrieving relevant document chunks from testprojectv2/ns1...")
    question = input["question"]
    chat_history_context = input["chat_history_context"]
    # logging.info("Retrieving relevant document chunks from testprojectv2/ns1...")
    # always include the question in []. because embed_documents expects a list. 
    # If its not a list, then each character will be treated as a separate document that needs to be embedded seperately.
    embeddedQuestion = st_model.encode([question + " " + chat_history_context], convert_to_numpy=True)[0].tolist()
    similar_docs = index.query(vector=embeddedQuestion, top_k=3, namespace="ns1", include_metadata=True)
    formatted_similar_docs = formatContext(similar_docs)
    return formatted_similar_docs


# Creating a method to format the retrieved elements(Q&A pairs or Document chunks) into a single string(For simplicity)
def formatContext(retrieved_elements):
    return "\n".join(doc.metadata["text"] for doc in retrieved_elements['matches'])


# Creating a prompt, which takes the context and question as inputs and passes them to the model
prompt = ChatPromptTemplate.from_template("""
    Answer the user question based on the following chathistory-based and knowledge-base context in no more than 150 characters.
    If you dont know the answer, just say you dont know.
    If you think the user question is not related to the knowledge base context, just ask the user to query on the topic of the knowledge base Indexed.
                                          
    Context: {context}
                                          
    Question: {question}""")


# Defining the LLM that will be used to answer the user query
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# Defining the output parser to parse the output of the LLM and exrtact the content out of the AIMessage object response
def outputParser(response):
    logging.info("RAG stage final: Parsing the response...")
    return response.content


# Wrap the 2 retrievers as runnables and creating a retriever_runnable
retrieveChatHistory_runnable = RunnableLambda(retrieve_chat_history)
retrieveRelevantDocs_runnable = RunnableLambda(retrieve_relevant_docs)
retriever_runnable = RunnableSequence(retrieveChatHistory_runnable, retrieveRelevantDocs_runnable)


# Chain for the retrieval and generation phases
rag_chain = (
    {"context": retriever_runnable, "question": RunnablePassthrough()}
    | prompt
    | model
    | outputParser
)
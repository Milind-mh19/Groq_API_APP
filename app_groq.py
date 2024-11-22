import streamlit as st
import os
from  langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

## load the api key 
os.environ['GROQ_API_KEY'] ="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"
os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provid context only.
    Please provide the most accurete respone based on the question 
    <context>
    {context}
    <context>
    Question:{input}
    """
)

st.title('MY RAG Application ')

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")## data ingestion stepst.
        st.session_state.docs=st.session_state.laoder.load()## document loding  
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=200)
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        


user_prompt =st.text_input("Enter your query form the reserch paper ")

if st.button("Document Emdedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriver=st.session_state.vectors.as_retriever()
    retriver_chain=create_retrieval_chain(retriver,document_chain)

    start =time.process_time()
    response=retriver_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write
    (response['answer'])
    ## With a streamlit expander  
    with st.expander("Document similarlity search"):
        for i,doc in enumerate(response['contrxt']):
            st.write(doc.page_content)
            st.write('------------------')

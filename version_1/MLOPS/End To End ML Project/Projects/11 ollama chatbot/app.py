import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful ai assistant. Please response answer based on question"),
    ("user","query:{query}")
])

model=ChatOllama(model="phi4:latest")
parser=StrOutputParser()
chain=prompt|model|parser

st.title("Chatbot using ollama")
st.sidebar.selectbox("Model",["phi4:latest"])
query=st.chat_input("Ask ")
response=chain.invoke(query)
st.write(response)
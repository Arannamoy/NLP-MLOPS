import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2']="true"

llm=ChatOllama(model="gpt-oss:latest")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert ML and AI engineer. You are also expert in data science and other tech field . Provide me answers based on the questions."),
        ("user","{input}")
    ]
)

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

query=st.chat_input("What questions you have ?")
response=chain.invoke({"input":query})

st.write(response)
print(response)

import streamlit as st
from langchain_community.llms.ollama import Ollama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=Ollama(model="qwen3:14b")

question=st.text_input("Enter your question:")
prompt=ChatPromptTemplate([
    f'{question}'
])

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if question:
   st.write(chain.invoke({"question":question}))


"""run command: streamlit run filename"""
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langserve import add_routes
load_dotenv()
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2']="true"
groq_api_key=os.getenv('GROQ_LPU_API')

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


prompt=ChatPromptTemplate([
    ('system','You are an expert ml engineer. Now answer based on the questions')
    ,('user',"{input}")
])

parser=StrOutputParser()

chain=prompt|llm|parser

# result=chain.invoke()


# app defination
app=FastAPI(title="Langchain Server",version="1.0",description="A simple API server using Langchain runnable interfaces")

add_routes(
    app,
    chain,
    path="/chain"
)


# if __name__=="__main__":
#     import uvicorn
#     uvicorn.run("main:app",host="localhost",port=8000,reload=True)
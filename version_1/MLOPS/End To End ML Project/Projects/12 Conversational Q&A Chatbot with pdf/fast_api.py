from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

app=FastAPI()
model=ChatOllama(model="phi4:latest")
def generate_prediction(query:str)->str:
   
    res=model.invoke(query)
    return res.content

@app.post("/{query}")
def prediction(query:str)->str:
    return generate_prediction(query)    

if __name__=="__main__":
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True)

from fastapi import requests,responses,FastAPI


app=FastAPI()

@app.get("/")
async def home():
    return {"messgae":"Hello"}
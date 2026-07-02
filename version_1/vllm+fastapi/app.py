from fastapi import FastAPI
import httpx
app=FastAPI()

@app.post("/:question:str")
async def questiion_answer(question:str):
    print(question)
    async with httpx.AsyncClient() as client:
        response = (await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "microsoft/Phi-4-mini-instruct",
                "messages": [{"role": "user", "content": question}]
            }
        )).json()
        print(response['id'])
        print(response['choices'][0]['message']['content'],type(response['choices'][0]['message']))
        return response['choices'][0]['message']['content']


    
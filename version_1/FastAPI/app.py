from fastapi import FastAPI, WebSocket, WebSocketDisconnect,Body,Path
from fastapi.responses import HTMLResponse
import ollama
import asyncio
import uvicorn
from books import BookRequest,BookModel

app = FastAPI()

books=[
    {"name":"Book1"},{"name":"Book2"},{"name":"Book3"},{"name":"Book4"}
]

# get methods

@app.get("/")
async def root():
    return "Hello World"

@app.get("/book")
async def get_all_book():
    return books

# Path Parameters
@app.get("/book/{param}")
async def specific_book(param=Path(max_length=2)):
    for book in books:
       if book.get('name').casefold()==param.casefold():
        print(param.casefold())
        return book
    print(param)
    return None


# Query Parameters
@app.get("/book/")
async def specific_book_query_param(name:str):
   for book in books:
       if book.get('name').casefold()==name.casefold():
        print(name.casefold())
        return book
   return None


# Post methods
@app.post("/create_book")
async def postBook(new_book:BookRequest):
    new_book=BookModel(**new_book.model_dump())
    books.append(new_book)
    print(new_book)
    return books


# Put methods
@app.put("/update_book")
async def updated_book(updated_book:BookRequest):
   print(updated_book)
   return books

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",reload=True)

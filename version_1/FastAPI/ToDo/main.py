from fastapi import FastAPI,Depends,Path,HTTPException,status
import models
from database import engine,SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
from models import Todos
from validation import TodoRequest
from routers import auth,todos
app=FastAPI()

models.Base.metadata.create_all(bind=engine)
app.include_router(auth.router)
app.include_router(todos.router)

# def get_db():
#     db=SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
# db_dependency=Annotated[Session,Depends(get_db)]

# @app.get("/")
# async def read_all(db:Annotated[Session,Depends(get_db)]):
#     return db.query(Todos).all()

# @app.get("/todo",status_code=status.HTTP_200_OK)
# async def get_todo(db:Annotated[Session,Depends(get_db)],todo_id:str):
#     todo=db.query(Todos).filter(Todos.id==todo_id).first()
#     print(todo)
#     if todo is not None:
#         return todo
#     raise HTTPException(status_code=404,detail="todo not found")

# @app.post("/create-todo",status_code=status.HTTP_201_CREATED)
# async def create_todo(db:Annotated[Session,Depends(get_db)],todo_request:TodoRequest):
#     new_todo=Todos(**todo_request.model_dump())
#     db.add(new_todo)
#     db.commit()
#     print(new_todo)


# @app.put("/update-todo/{todo_id}",status_code=status.HTTP_200_OK)
# async def update_todo(db:Annotated[Session,Depends(get_db)],todo_request:TodoRequest,todo_id:int):
#     print(todo_id)
#     todo_model=db.query(Todos).filter(Todos.id==todo_id).first()
#     todo_request=Todos(**todo_request.model_dump())
#     print(todo_model,todo_request)
#     if todo_model is None:
#         raise HTTPException(status_code=404,detail="todo not found")
#     todo_model.title=todo_request.title
#     todo_model.description=todo_request.description
#     todo_model.priority=todo_request.priority
#     todo_model.complete=todo_request.complete
#     db.add(todo_model)
#     db.commit()
#     return todo_model


# @app.delete("/delete-todo/{todo_id}",status_code=status.HTTP_200_OK)
# async def delete_todo(db:db_dependency,todo_id:int):
#     db.query(Todos).filter(Todos.id==todo_id).delete()
#     db.commit()



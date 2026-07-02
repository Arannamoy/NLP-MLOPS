from pydantic import BaseModel,Field
from typing import Optional

class TodoRequest(BaseModel):
    title:str=Field()
    description:str=Field()
    priority:str=Field()
    complete:Optional[bool]=Field(default=False)
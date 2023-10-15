from pydantic import BaseModel
from typing import Optional

# | = opcional porque mongodb lo inserta automaticamente, si lo hace
class Documento(BaseModel):
    id: str | None = None 
    name: str
    name_document: str
    date: str
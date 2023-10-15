from pydantic import BaseModel
from typing import Optional

# | = opcional porque mongodb lo inserta automaticamente, si lo hace
class Url(BaseModel):
    # id: Optional[str]  
    name: str
    url: str
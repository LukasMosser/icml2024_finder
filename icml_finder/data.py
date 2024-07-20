from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from datetime import datetime


# Define the Pydantic data model
class Session(BaseModel):
    type: str
    name: str
    virtualsite_url: str
    speakers_authors: str
    abstract: str
    location: Optional[str] = None
    time_vienna: Optional[datetime] = None
    embedding:  Optional[List[float]] = None






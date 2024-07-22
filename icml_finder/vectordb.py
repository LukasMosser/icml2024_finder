import lancedb
from icml_finder.data import Session
import os
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector

embeddings = get_registry().get("openai").create(name="text-embedding-3-large")

class LanceSchema(LanceModel):
    embedding: Vector(embeddings.ndims()) = embeddings.VectorField()
    abstract: str = embeddings.SourceField()
    payload: Session

    class Config:
        frozen = True

def make_vectordb(vectordb_dir):
    if os.path.exists(vectordb_dir):
        db = lancedb.connect(vectordb_dir)
        return db.open_table("icml2024")
    else:
        st.error("Could not connect to Vector Database")

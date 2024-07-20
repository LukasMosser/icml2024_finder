import lancedb
import pandas as pd
from icml_finder.data import Session
import jsonlines 
import json 
import os
from shutil import rmtree
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector, LanceModel
from lancedb.rerankers import CohereReranker

embeddings = get_registry().get("openai").create(name="text-embedding-3-large")

class LanceSchema(LanceModel):
    embedding: Vector(embeddings.ndims()) = embeddings.VectorField()
    abstract: str = embeddings.SourceField()
    payload: Session

    class Config:
            frozen = True

def make_vectordb(input_file_path, vectordb_dir):

    if not os.path.exists(vectordb_dir):
        db = lancedb.connect(vectordb_dir)

        table_name = "icml2024"
        table = db.create_table(table_name, schema=LanceSchema)

        with jsonlines.open(input_file_path) as reader:
            lance_objs = []
            for i, obj in enumerate(reader):
                session = json.loads(obj) 
                embedding = session['embedding'].copy()
                session['embedding'] = None
                
                session_obj = Session(**session)
                lance_obj = LanceSchema(embedding=embedding, abstract=session_obj.abstract, payload=session_obj)
                lance_objs.append(lance_obj)

            table.add(lance_objs)

        table.create_fts_index(["abstract"], replace=True)

        return table
    else:
        db = lancedb.connect(vectordb_dir)
        return db.open_table("icml2024")
import os
import jsonlines 
import json 
import lancedb 
from lancedb.embeddings import get_registry
from shutil import rmtree
from icml_finder.data import Session
from icml_finder.vectordb import LanceSchema

embeddings = get_registry().get("openai").create(name="text-embedding-3-large")

def make_vectordb(input_file_path: str, vectordb_dir: str):
    if os.path.exists(vectordb_dir):
        rmtree(vectordb_dir)
    db = lancedb.connect(vectordb_dir)

    table_name = "icml2024"
    table = db.create_table(table_name, schema=LanceSchema)

    with jsonlines.open(input_file_path) as reader:
        lance_objs = []
        for i, obj in enumerate(reader):
            session = json.loads(obj)
            embedding = session["embedding"].copy()
            session["embedding"] = None

            session_obj = Session(**session)
            lance_obj = LanceSchema(
                embedding=embedding,
                abstract=session_obj.abstract,
                payload=session_obj,
            )
            lance_objs.append(lance_obj)

        table.add(lance_objs)

    table.create_fts_index(["abstract"], replace=True)

if __name__ == "__main__":
    make_vectordb("./data/icml_sessions.jsonl", "./data/vectordb")
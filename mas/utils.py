from sentence_transformers import SentenceTransformer
import yaml
import os
from typing import Union, Any
import random
import json
from dataclasses import dataclass
import math


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_json(file_name: str) -> Union[list, dict]:

    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, separators=(",", ": "))

def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """
    Divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.
    """
    if len(lst) == 0:
        return []
    
    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    else:
        num_chunks = math.ceil(len(lst) / k)
        chunk_size = math.ceil(len(lst) / num_chunks)
        return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    

_EMBEDDING_MODEL_CACHE = {} 

@dataclass
class EmbeddingFunc:

    model_type: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        if self.model_type not in _EMBEDDING_MODEL_CACHE:
            _EMBEDDING_MODEL_CACHE[self.model_type] = SentenceTransformer(self.model_type)

        self.func: SentenceTransformer = _EMBEDDING_MODEL_CACHE[self.model_type]

    def embed_documents(self, texts: list[str]) -> list[list]:
        return [self.func.encode(text).tolist() for text in texts]

    def embed_query(self, query: str) -> list:
        return self.func.encode(query).tolist()



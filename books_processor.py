from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import weaviate

import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Property, DataType

from weaviate.collections import Collection
from weaviate.collections.classes.config import (
    Property, DataType
)

from enum import Enum
from typing import Dict
import numpy as np
from math import floor
from typing import List, Dict, Optional
from llmlingua import PromptCompressor
from jinja2 import Template
import dotenv
import os

dotenv.load_dotenv()
llm_name = os.getenv("LLM")
prompts_folder = os.getenv("PROMPTS_FOLDER")
embedding_model_path = os.getenv("ENCODER_MODEL")

embedding_model = SentenceTransformer(embedding_model_path, trust_remote_code=True, device='cuda')
compressor = PromptCompressor(model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank', use_llmlingua2=True)
#wv_client = weaviate.connect_to_local()
wv_client = weaviate.connect_to_local(
    host='81.94.156.34',  # Укажите адрес хоста
    port=8080,         # Укажите порт HTTP
    grpc_port=50051    # Укажите порт gRPC
)
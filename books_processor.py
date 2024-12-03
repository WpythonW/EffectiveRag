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
    host=os.getenv("REMOTE_SERVER_IP"),  # Укажите адрес хоста
    port=os.getenv("WEAVIATE_PORT_REST"),         # Укажите порт HTTP
    grpc_port=os.getenv("WEAVIATE_PORT_GRPC")    # Укажите порт gRPC
)

class BooksProcessor:
    def __init__(self, wv_client, embedding_model):
        self.embedding_model = embedding_model
        self.wv_client = wv_client

    def create_collection_if_not_exists(self, collection_name):
        if not self.wv_client.collections.exists(collection_name):
            self.wv_client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="chunk", data_type=DataType.TEXT),
                    Property(name="book_name", data_type=DataType.TEXT),
                    Property(name="chunk_num", data_type=DataType.INT)
                ],
            )
        return self.wv_client.collections.get(collection_name)

    def split_book(self, book_text, chunk_size, chunk_overlap):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return [i.page_content for i in splitter.create_documents([book_text])]

    def process_book(self, book_name, book_txt):
        if self.wv_client.collections.exists(book_name + '_medium_chunks'):
            print("Book already exists")
            return
        chunk_configs = [
        #    ('_big_chunks', 3000, 1000),
            ('_medium_chunks', 1000, 100),
        #    ('_small_chunks', 400, 50)
        ]
        
        for suffix, chunk_size, overlap in chunk_configs:
            collection = self.create_collection_if_not_exists(book_name + suffix)
            chunks = self.split_book(book_txt, chunk_size, overlap)
            embeddings = self.embedding_model.encode(['search_document: ' + i for i in chunks], batch_size=15).tolist()
            question_objs = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                question_objs.append(wvc.data.DataObject(
                    properties= {
                        "chunk": chunk,
                        "book_name": book_name,
                        "chunk_num": i
                    },
                    vector=embedding
                ))
            collection.data.insert_many(question_objs)

    def delete_book(self, book_name: str) -> None:
        """
        Delete all collections associated with a book.
        """
        for suffix in ['_big_chunks', '_medium_chunks', '_small_chunks']:
            collection_name = book_name + suffix
            if self.wv_client.collections.exists(collection_name):
                try:
                    self.wv_client.collections.delete(collection_name)
                except Exception as e:
                    print(f"Error deleting collection {collection_name}: {e}")
        print(f"Successfully deleted collections for {book_name}")

class Search:
    def __init__(self, wv_client, embedding_model):
        self.embedding_model = embedding_model
        self.wv_client = wv_client
        self.multiplier_mapping = {'_big_chunks': 0.7, '_medium_chunks': 1, '_small_chunks': 1.9}
        #self._load_prompt_template()

    def search(self, query, book_name):
        collection_type = '_medium_chunks'
        print(f'Collection type: {collection_type}')
        book = self.wv_client.collections.get(book_name + collection_type)
        
        total_count = book.aggregate.over_all(total_count=True).total_count
        chunks_to_retrieve = floor(np.maximum(self.multiplier_mapping[collection_type] * np.log(total_count), 1))
        print(f"Retrieving {chunks_to_retrieve} chunks")
        
        embedding = self.embedding_model.encode('search_query: ' + query, batch_size=1)
        response = book.query.near_vector(near_vector=list(embedding), limit=chunks_to_retrieve, return_metadata=wvc.query.MetadataQuery(certainty=True))
        relevant_chunks = response.objects#sorted(response.objects, key=lambda x: x.properties['chunk_num'])
        relevant_text = '\n'.join([f'\nCHUNK {i.properties['chunk_num']}\n' + i.properties['chunk'].strip() for i in relevant_chunks])
        print(f'Len of relevant text: {len(relevant_text)}')
        return relevant_text

class RAGSystem:
    def __init__(self, wv_client, embedding_model, compressor, llm_name, prompts_folder, compression_rate=0.75):
        self.embedding_model = embedding_model
        self.searcher = Search(wv_client, self.embedding_model)
        self.compression_rate = compression_rate
        self.compressor = compressor
        self.llm = OllamaLLM(
            model=llm_name,
            temperature=0,
            base_url=f"http://{os.getenv("REMOTE_SERVER_IP")}:{os.getenv("OLLAMA_PORT")}"
        )
        with open(os.path.join(prompts_folder, 'final_prompt.j2')) as f:
            self._template = f.read()

    def query(self, query: str, book_names: List[str], 
             dialogue_history: Optional[List[Dict[str, str]]] = None) -> str:
        dialogue_history = dialogue_history or []
        compressed_contexts = []
        
        for book_name in book_names:
            context = self.searcher.search(query, book_name)
            if context:
                compressed = self.compressor.compress_prompt(
                    context,
                    rate=self.compression_rate,
                    force_tokens=['\n', '?', '.', '!', 'CHUNK']
                )['compressed_prompt']
                compressed_contexts.append(f"From {book_name}:\n{compressed}")
        
        if not compressed_contexts:
            return "No relevant information found."

        print(f'Len of compressed context: {sum([len(i) for i in compressed_contexts])}')
        final_prompt = Template(self._template).render(
            contexts=compressed_contexts,
            dialogue_history=dialogue_history,
            query=query
        )
        
        return self.llm.invoke(final_prompt)
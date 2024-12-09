import streamlit as st
from streamlit_chat import message
from st_clickable_images import clickable_images
import random
import os
import base64
import dotenv
import weaviate
from weaviate.classes import Property, DataType
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llmlingua import PromptCompressor
from jinja2 import Template

dotenv.load_dotenv()
llm_name = os.getenv("LLM")
prompts_folder = os.getenv("PROMPTS_FOLDER")
embedding_model_path = os.getenv("ENCODER_MODEL")

embedding_model = SentenceTransformer(embedding_model_path, trust_remote_code=True, device='cuda')
compressor = PromptCompressor(model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank', use_llmlingua2=True)
wv_client = weaviate.connect_to_local(
    host='81.94.156.34',
    port=8080,
    grpc_port=50051
)
llm = OllamaLLM(model=llm_name, temperature=0, base_url="http://localhost:11434")

with open(os.path.join(prompts_folder, 'final_prompt.j2')) as f:
    final_prompt_template = f.read()


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
        #   ('_big_chunks', 3000, 1000),
            ('_medium_chunks', 1000, 100),
        #   ('_small_chunks', 400, 50)
        ]

        for suffix, chunk_size, overlap in chunk_configs:
            collection = self.create_collection_if_not_exists(book_name + suffix)
            chunks = self.split_book(book_txt, chunk_size, overlap)
            embeddings = self.embedding_model.encode(['search_document: ' + i for i in chunks], batch_size=15).tolist()
            question_objs = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                question_objs.append(wvc.data.DataObject(
                    properties={
                        "chunk": chunk,
                        "book_name": book_name,
                        "chunk_num": i
                    },
                    vector=embedding
                ))
            collection.data.insert_many(question_objs)

processor = BooksProcessor(wv_client, embedding_model)


class Search:
    def __init__(self, wv_client, embedding_model):
        self.embedding_model = embedding_model
        self.wv_client = wv_client
        self.multiplier_mapping = {'_medium_chunks': 1}

    def search(self, query, book_name):
        collection_type = '_medium_chunks'
        book = self.wv_client.collections.get(book_name + collection_type)

        total_count = book.aggregate.over_all(total_count=True).total_count
        chunks_to_retrieve = floor(np.maximum(self.multiplier_mapping[collection_type] * np.log(total_count), 1))

        embedding = self.embedding_model.encode('search_query: ' + query, batch_size=1)
        response = book.query.near_vector(near_vector=list(embedding), limit=chunks_to_retrieve, return_metadata=wvc.query.MetadataQuery(certainty=True))
        relevant_chunks = response.objects
        relevant_text = '\n'.join([f'\nCHUNK {i.properties["chunk_num"]}\n' + i.properties['chunk'].strip() for i in relevant_chunks])
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
            base_url="http://localhost:11434"
        )
        self.final_prompt_template = final_prompt_template

    def query(self, query: str, book_names: List[str], dialogue_history: Optional[List[Dict[str, str]]] = None) -> str:
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

        final_prompt = Template(self.final_prompt_template).render(
            contexts=compressed_contexts,
            dialogue_history=dialogue_history,
            query=query
        )

        return self.llm.invoke(final_prompt)

# Инициализация RAG системы
rag = RAGSystem(wv_client, embedding_model, compressor, llm_name, prompts_folder)


# Интеграция с интерфейсом Streamlit

# Заголовок ко всей работе
st.title("ITMO команда BookWise")

# Вставка изображения
st.image("wise.webp", caption="BookWise Logo", use_container_width=True, width=256)

# Интерфейс Streamlit
st.header("Book RAG Chatbot")

# Функция для обработки ввода пользователя
def on_input_change():
    user_input = st.session_state.user_input
    if user_input:
        response = rag.query(query=user_input, book_names=[st.session_state.selected_book_name])
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        st.session_state.user_input = ""  # Очистка поля ввода

# Функция для очистки истории чата
def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

# Инициализация состояния сессии
if 'past' not in st.session_state:
    st.session_state.past = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'book_images' not in st.session_state:
    st.session_state.book_images = []
if 'selected_book_index' not in st.session_state:
    st.session_state.selected_book_index = 0

# Блок чата
chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i], is_user=True, key=f"{i}_user")
        message(st.session_state.generated[i], key=f"{i}")

    st.button("Очистить чат", on_click=on_btn_click)

with st.container():
    st.text_input("Введите ваш вопрос:", on_change=on_input_change, key="user_input")

# Блок 4: Загрузка файла
st.header("Загрузите книгу (.txt):")
uploaded_file = st.file_uploader("Выберите файл", type="txt")

if uploaded_file is not None:
    # Чтение файла
    with uploaded_file as file:
        content = file.read().decode("utf-8")

    # Отображение имени файла
    st.write(f"Загруженный файл: {uploaded_file.name}")

    # Отображение первых 10 строк файла
    st.header("Превью файла:")
    preview_lines = content.split('\n')[:10]
    preview_text = '\n'.join(preview_lines)
    st.text(preview_text)

    # Добавление изображения книги
    book_name = os.path.splitext(uploaded_file.name)[0]
    image_path = f"{book_name}.jpeg"
    if os.path.exists(image_path) and image_path not in st.session_state.book_images:
        with open(image_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
            st.session_state.book_images.append(f"data:image/jpeg;base64,{encoded}")
        st.session_state.selected_book_index = len(st.session_state.book_images) - 1

    # Обработка книги
    processor.process_book(book_name, content)

# Блок выбора книги через изображение
if st.session_state.book_images:
    clicked = clickable_images(
        st.session_state.book_images,
        titles=[f"Книга #{str(i)}" for i in range(len(st.session_state.book_images))],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )
    if clicked > -1 and clicked != st.session_state.selected_book_index:
        st.session_state.selected_book_index = clicked
        on_btn_click()  # Очистка истории чата при выборе новой книги
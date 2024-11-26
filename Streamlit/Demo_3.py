import streamlit as st
from streamlit_chat import message
from streamlit_image_select import image_select
import random
import os

# Списки заглушек для ответов
responses = [
    "Это интересный вопрос!",
    "Пожалуйста, уточните ваш запрос.",
    "Ответ на ваш вопрос находится в главе 3.",
    "Этот персонаж появляется в главе 5.",
    "Сюжет развивается в главе 7.",
]

responses_2 = [
    "Это другая книга!",
    "Пожалуйста, уточните ваш запрос для этой книги.",
    "Ответ на ваш вопрос находится в главе 4.",
    "Этот персонаж появляется в главе 6.",
    "Сюжет развивается в главе 8.",
]

# Функция для генерации случайного ответа
def get_random_response(book_index):
    if book_index == 0:
        return random.choice(responses)
    elif book_index == 1:
        return random.choice(responses_2)
    else:
        return "Неизвестная книга."

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
        response = get_random_response(st.session_state.selected_book_index)
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
    image_path = f"{book_name}.jpg"
    if os.path.exists(image_path):
        st.session_state.book_images.append(image_path)
        st.session_state.selected_book_index = len(st.session_state.book_images) - 1

# Блок выбора книги через изображение
if st.session_state.book_images:
    selected_image = image_select("Выберите книгу:", st.session_state.book_images, index=st.session_state.selected_book_index)
    if selected_image:
        st.session_state.selected_book_index = st.session_state.book_images.index(selected_image)
        on_btn_click()  # Очистка истории чата при выборе новой книги
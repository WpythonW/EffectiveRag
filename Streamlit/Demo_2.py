import streamlit as st
from streamlit_chat import message
import random

# Список заглушек для ответов
responses = [
    "Это интересный вопрос!",
    "Пожалуйста, уточните ваш запрос.",
    "Ответ на ваш вопрос находится в главе 3.",
    "Этот персонаж появляется в главе 5.",
    "Сюжет развивается в главе 7.",
]

# Функция для генерации случайного ответа
def get_random_response():
    return random.choice(responses)

# Заголовок ко всей работе
st.title("ITMO команда BookWise")

# Вставка изображения
st.image("wise.webp", use_container_width=True, width=256)

# Интерфейс Streamlit
st.header("Book RAG Chatbot")

# Функция для обработки ввода пользователя
def on_input_change():
    user_input = st.session_state.user_input
    if user_input:
        response = get_random_response()
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
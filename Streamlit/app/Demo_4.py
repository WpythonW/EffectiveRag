import streamlit as st
from st_clickable_images import clickable_images
import random
import os
import base64

# Список заглушек для ответов
responses = {
    "default.txt": [
        "Это интересный вопрос!",
        "Пожалуйста, уточните ваш запрос.",
        "Ответ на ваш вопрос находится в главе 3.",
        "Этот персонаж появляется в главе 5.",
        "Сюжет развивается в главе 7.",
    ],
    "Sherlock_Study_in_Scarlet.txt": [
        "Это интересный вопрос!",
        "Пожалуйста, уточните ваш запрос.",
        "Ответ на ваш вопрос находится в главе 3.",
        "Этот персонаж появляется в главе 5.",
        "Сюжет развивается в главе 7.",
    ],
    "Another_Book.txt": [
        "Ответ на ваш вопрос находится в главе 2.",
        "Этот персонаж появляется в главе 4.",
        "Сюжет развивается в главе 6.",
    ],
    # Добавьте другие книги и соответствующие ответы
}

# Функция для генерации случайного ответа
def get_random_response(book_name):
    return random.choice(responses[book_name])

# Заголовок ко всей работе
st.title("ITMO команда BookWise")

# Вставка изображения
st.image("wise.webp", use_container_width=True, width=128)

# Интерфейс Streamlit
st.header("Book RAG Chatbot")

# Инициализация состояния сессии
if 'past' not in st.session_state:
    st.session_state.past = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'current_book' not in st.session_state:
    st.session_state.current_book = "default.txt"
if 'images' not in st.session_state:
    st.session_state.images = []
if 'books' not in st.session_state:
    st.session_state.books = []

# Проверка наличия начального изображения
default_image_path = "images/choice.jpeg"
if os.path.exists(default_image_path):
    with open(default_image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        st.session_state.images.append(f"data:image/jpeg;base64,{encoded}")

# Блок 4: Загрузка файла
st.header("Загрузите книгу (.txt):")
uploaded_file = st.file_uploader("Выберите файл", type="txt")

if uploaded_file is not None:
    # Чтение файла
    content = uploaded_file.read().decode("utf-8")

    # Отображение имени файла
    st.write(f"Загруженный файл: {uploaded_file.name}")

    # Отображение первых 10 строк файла
    st.header("Превью файла:")
    preview_lines = content.split('\n')[:10]
    preview_text = '\n'.join(preview_lines)
    st.text(preview_text)

    # Сохранение файла и изображения
    file_name = uploaded_file.name
    file_path = f"books/{file_name}"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Предполагаем, что изображение с таким же именем существует
    image_name = file_name.replace(".txt", ".jpeg")
    image_path = f"images/{image_name}"

    # Добавляем изображение книги в список изображений, если его там еще нет
    if image_path not in st.session_state.images:
        with open(image_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
            st.session_state.images.append(f"data:image/jpeg;base64,{encoded}")

    # Добавление ответов для новой книги
    responses[file_name] = [
        f"Ответ на ваш вопрос находится в главе {i+1}." for i in range(5)
    ]

    # Добавление книги в список загруженных книг
    if file_name not in st.session_state.books:
        st.session_state.books.append(file_name)

# Выбор изображения
st.header("Выберите книгу:")
clicked = clickable_images(
    st.session_state.images,
    titles=[f"Image #{str(i)}" for i in range(len(st.session_state.images))],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)

# Обновление текущей книги
if clicked > -1:
    st.session_state.current_book = os.path.basename(st.session_state.images[clicked]).replace(".jpeg", ".txt")
    st.session_state.past = []
    st.session_state.generated = []

# Проверка, загружена ли книга
if not st.session_state.books:
    st.warning("Пожалуйста, загрузите книгу.")
else:
    # Функция для обработки ввода пользователя
    def on_input_change():
        user_input = st.session_state.user_input
        if user_input:
            try:
                response = get_random_response(st.session_state.current_book)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
                st.session_state.user_input = ""  # Очистка поля ввода
            except KeyError:
                st.error("Книга не найдена. Пожалуйста, выберите другую книгу.")

    # Функция для очистки истории чата
    def on_btn_click():
        st.session_state.past = []
        st.session_state.generated = []

    # Блок чата
    st.header("Чат")
    for i in range(len(st.session_state.generated)):
        st.write(f"**Вы:** {st.session_state.past[i]}")
        st.write(f"**Бот:** {st.session_state.generated[i]}")

    st.button("Очистить чат", on_click=on_btn_click)

    with st.container():
        st.text_input("Введите ваш вопрос:", on_change=on_input_change, key="user_input")
import streamlit as st
import random

# Список заглушек для ответов
responses = [
    "Это интересный вопрос!",
    "Это было в Лондоне",
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
# st.image("wise.webp", caption="BookWise Logo", use_container_width=True, width=128)
st.image("wise.webp", use_container_width=True, width=128)

# Интерфейс Streamlit
st.header("Book RAG Chatbot")

# Блок 1: Ввод запроса
st.header("Введите ваш вопрос:")
query = st.text_input("Запрос:")

# Блок 2: Вывод ответа
st.header("Ответ:")
if st.button("Отправить"):
    if query:
        response = get_random_response()
        st.write(response)

        # Сохранение истории запросов и ответов
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"query": query, "response": response})
    else:
        st.write("Пожалуйста, введите вопрос.")

# Блок 3: История запросов
st.sidebar.title("История запросов")
if 'history' in st.session_state:
    for item in st.session_state.history:
        st.sidebar.write(f"Вопрос: {item['query']}")
        st.sidebar.write(f"Ответ: {item['response']}")
        st.sidebar.write("---")


# Блок 4: Загрузка файла
st.header("Загрузите книгу (.txt):")
uploaded_file = st.file_uploader("Выберите файл", type="txt")

if uploaded_file is not None:    # Чтение файла
    with uploaded_file as file:
        content = file.read().decode("utf-8")

    # Отображение имени файла
    st.write(f"Загруженный файл: {uploaded_file.name}")

    # Отображение первых 10 строк файла
    st.header("Превью файла:")
    preview_lines = content.split('\n')[:10]
    preview_text = '\n'.join(preview_lines)
    st.text(preview_text)
# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем необходимые зависимости
RUN pip install streamlit st-clickable-images

# Копируем ваш код в контейнер
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда для запуска Streamlit
CMD ["streamlit", "run", "Demo_4.py"]
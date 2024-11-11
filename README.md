# Установка и запуск Ollama с Weaviate

Приложение из двух контейнеров(ollama и weaviate). Weaviate хостит векторную бд и поддерживает различные поисковые алгоритмы. Ollama предлагает две модели - LLM llama3.2-3B и all-minilm:33m, которые можно использовать напрямую с помощью langchain или внутри weaviate для векторизации игенерации.

## Запуск на CPU

Для запуска на CPU используйте файл `docker-compose-cpu.yml`:

```bash
docker compose -f docker-compose-cpu.yml up --build
```

## Запуск на GPU (NVIDIA)

### 1. Установка NVIDIA Container Toolkit

#### Для систем с apt (Ubuntu/Debian):

```bash
# Добавление ключа и репозитория
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Обновление пакетов
sudo apt-get update

# Установка NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
```

#### Для систем с yum/dnf (CentOS/RHEL/Fedora):

```bash
# Добавление репозитория
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Установка NVIDIA Container Toolkit
sudo yum install -y nvidia-container-toolkit
```

### 2. Настройка Docker для работы с NVIDIA

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Запуск с GPU

```bash
docker compose -f docker-compose-gpu.yml up --build
```

## Примечания

- В `.env` файле вы можете изменить пути к папкам с данными и модели по умолчанию
- Перед запуском убедитесь, что указанные в `.env` директории существуют и у Docker есть права на запись в них
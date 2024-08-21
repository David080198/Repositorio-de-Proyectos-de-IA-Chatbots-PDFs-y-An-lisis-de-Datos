from python:latest

WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    espeak \
    libespeak1 \
    && rm -rf /var/lib/apt/lists/*

COPY requeriments.txt ./
COPY app.py ./

RUN pip install --upgrade pip
RUN pip install -r requeriments.txt

EXPOSE 8501
ENTRYPOINT [ "streamlit","run", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

RUN mkdir -p /app/data/uploads \
    && mkdir -p /app/backend/vector_store/faiss_index

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "frontend/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
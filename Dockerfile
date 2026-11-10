FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DL_LAB_SERVE_FRONTEND=1

WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
# O laboratório é executado em CPU no B1. Usar o índice CPU evita baixar
# bibliotecas CUDA desnecessárias e deixa a imagem adequada ao plano de aula.
RUN grep -vE '^(torch|torchvision)>=' /tmp/requirements.txt > /tmp/base-requirements.txt \
    && pip install --no-cache-dir -r /tmp/base-requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY backend /app/backend
COPY frontend/dist /app/frontend/dist
RUN mkdir -p /app/backend/data

WORKDIR /app/backend
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

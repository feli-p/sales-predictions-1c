FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Instalando uv
RUN pip install --no-cache-dir uv


COPY pyproject.toml uv.lock /app/


RUN uv sync --frozen --no-dev


COPY . /app

ENV PYTHONPATH=/app

ENTRYPOINT ["uv", "run", "python", "-m", "src.preprocessing"]
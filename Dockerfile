FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md .
COPY src ./src
COPY app ./app
COPY data ./data

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"]

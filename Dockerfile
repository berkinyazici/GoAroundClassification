FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/best_model.joblib

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN mkdir -p data/raw data/interim data/processed models reports/metrics reports/figures \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

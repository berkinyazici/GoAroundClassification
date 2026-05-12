FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    polars pandas pyarrow scikit-learn imbalanced-learn lightgbm \
    fastapi "uvicorn[standard]" jinja2 python-multipart joblib \
    matplotlib seaborn tqdm requests numpy

# Copy project source
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY reports/ ./reports/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

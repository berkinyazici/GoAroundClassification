FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY web /app/web
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "goaround.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

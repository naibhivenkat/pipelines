FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r pyproject.toml
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8081"]

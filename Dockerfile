# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Create working directory
WORKDIR /app

# Copy only dependency declarations to install first (cache efficient)
COPY pyproject.toml poetry.lock /app/

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Copy rest of the code
COPY . /app

# Expose default port (can be adjusted for K8s)
EXPOSE 8000

# Run the app (adjust as needed)
CMD ["python", "example.py"]

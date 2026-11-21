FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 LEARNER_ADAPTERS_ENABLED=true LEARNER_ADAPTER_ROOT=/app/models/adapters
EXPOSE 10000
CMD ["sh", "-c", "uvicorn insight_learning.api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]

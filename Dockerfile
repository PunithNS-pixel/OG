FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate datasets at build time so first reset() is instant
RUN python -m data.generate

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "8000"]

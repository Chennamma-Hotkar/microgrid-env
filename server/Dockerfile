FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . .

# Install Python package
RUN pip install --no-cache-dir -e .

# HF Spaces uses port 7860
EXPOSE 7860

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
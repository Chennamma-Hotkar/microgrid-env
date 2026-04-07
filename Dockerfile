FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic websockets httpx openenv-core

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

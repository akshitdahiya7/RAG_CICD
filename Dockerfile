FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Cache mount persists pip's download cache across interrupted/retried builds
# (a killed RUN step isn't cached at all otherwise, forcing a full re-download
# every retry) — this is a genuinely large dependency set (torch, ragas, mlflow).
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]

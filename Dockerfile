FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts
COPY outputs ./outputs
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "favorita_forecasting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

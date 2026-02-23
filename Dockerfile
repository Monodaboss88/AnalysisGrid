FROM python:3.11-slim

WORKDIR /app

# Install system deps for numpy/pandas C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Railway injects $PORT at runtime (typically 3000-9000)
ENV PORT=8000
EXPOSE ${PORT}

# MUST use shell form so $PORT is expanded at runtime
CMD python -m uvicorn app:app --host 0.0.0.0 --port $PORT

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY *.html ./

# Create data directory
RUN mkdir -p scanner_data

# Expose port
EXPOSE 8000

# Run the server - use PORT env variable from Railway
CMD python -m uvicorn unified_server:app --host 0.0.0.0 --port ${PORT:-8000}

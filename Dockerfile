# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Expose port (HF listens on $PORT)
ENV PORT 7860
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]

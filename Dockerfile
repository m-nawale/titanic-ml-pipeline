# Base: small, recent Python
FROM python:3.11-slim

# Non-interactive & faster Python/pip defaults
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System build tools (needed for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside container
WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create non-root user so files created match uid 1000 (common on WSL)
RUN useradd -ms /bin/bash -u 1000 appuser
USER appuser

# Default workdir for bind-mounts
WORKDIR /app

# Streamlit port (optional)
EXPOSE 8501

# Default command; we’ll override when running
CMD ["python", "--version"]

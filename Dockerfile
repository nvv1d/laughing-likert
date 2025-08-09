# 1. Base image: Python
FROM python:3.11-slim

# 2. System dependencies for Python scientific stack
RUN echo 'Acquire::Retries "5"; Acquire::http::Timeout "60"; Acquire::https::Timeout "60";' \
      > /etc/apt/apt.conf.d/99timeouts \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential \
       zlib1g-dev libbz2-dev liblzma-dev libpcre2-dev libdeflate-dev libzstd-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Setup Python venv and install Python dependencies
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY . /app

# 5. Expose and start Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

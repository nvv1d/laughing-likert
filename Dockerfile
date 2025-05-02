FROM python:3.11-slim

WORKDIR /app

# 1. Install system deps for R + compilers for building packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    r-base \
    r-base-dev \
 && rm -rf /var/lib/apt/lists/*

# 2. Confirm Rscript is present (will fail build if it's missing)
RUN which Rscript && Rscript --version

# 3. Pre-install CRAN packages you need
RUN Rscript -e "install.packages(c('psych','lavaan'), repos='https://cloud.r-project.org/')"

# 4. Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Layout
RUN mkdir -p ./data ./templates

COPY app.py .
COPY utils.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY likert_pattern_analysis.py .
COPY templates/ ./templates/
COPY data/ ./data/

# 6. Streamlit config
RUN mkdir -p ./.streamlit \
 && printf "[server]\nheadless = true\nport = 5000\nenableCORS = false\nenableXsrfProtection = true\n" \
    > ./.streamlit/config.toml

EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]

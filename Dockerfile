# 1. Base image
FROM python:3.11-slim@sha256:75a17dd6f00b277975715fc094c4a1570d512708de6bb4c5dc130814813ebfe4

# 2. Install system dependencies & R
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

# 3. Set working directory
WORKDIR /app

# 4. Install R packages
RUN Rscript -e "install.packages(c('psych','lavaan'), repos='https://cloud.r-project.org/')"

# 5. (Optional) Verify R installation
RUN which Rscript && Rscript --version

# 6. Copy Python requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code
COPY app.py .
COPY likert_pattern_analysis.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY utils.py .

# 8. Copy your templates folder
COPY templates/ ./templates/

# 9. Expose port and default command (adjust if needed)
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

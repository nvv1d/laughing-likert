# Use Python slim image with R installation
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure R
RUN mkdir -p /usr/lib/R/etc && \
    echo 'options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4, timeout = 600)' > /usr/lib/R/etc/Rprofile.site

# Working directory
WORKDIR /app

# Install required R packages one by one
RUN Rscript -e "install.packages('readr')"
RUN Rscript -e "install.packages('readxl')"
RUN Rscript -e "install.packages('polycor')"
RUN Rscript -e "install.packages('psych')"
RUN Rscript -e "install.packages('lavaan')"
RUN Rscript -e "install.packages('simstudy')"
RUN Rscript -e "install.packages('mokken')"
RUN Rscript -e "install.packages('semTools')"
RUN Rscript -e "install.packages('lordif')"

# Verify R package installation
RUN Rscript -e "installed_pkgs <- installed.packages()[,'Package']; for(pkg in c('readr', 'readxl', 'polycor', 'psych', 'simstudy', 'mokken', 'lavaan', 'semTools', 'lordif')) { cat(pkg, ifelse(pkg %in% installed_pkgs, 'installed', 'NOT INSTALLED'), '\n') }"

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY likert_pattern_analysis.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY utils.py .

# Copy templates
COPY templates/ ./templates/

# Expose port and start command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

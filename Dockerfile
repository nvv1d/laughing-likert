# Use Python slim image with R installation
FROM python:3.11-slim

# Install system dependencies in a single layer to optimize build
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure R with faster CRAN mirror and more CPUs
RUN mkdir -p /usr/lib/R/etc && \
    echo 'options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4, timeout = 600)' > /usr/lib/R/etc/Rprofile.site

# Working directory
WORKDIR /app

# Copy requirements first to leverage layer caching
COPY requirements.txt .

# Install Python requirements - do this before R packages to save time if Python build fails
RUN pip install --no-cache-dir -r requirements.txt

# Install all R packages in one command to reduce layers and time
RUN Rscript -e "\
    packages <- c('readr', 'readxl', 'polycor', 'psych', 'lavaan', 'simstudy', 'mokken', 'semTools', 'lordif'); \
    new_packages <- packages[!(packages %in% installed.packages()[,'Package'])]; \
    if(length(new_packages) > 0) { \
        install.packages(new_packages, dependencies=TRUE); \
    }; \
    installed_pkgs <- installed.packages()[,'Package']; \
    for(pkg in packages) { \
        cat(pkg, ifelse(pkg %in% installed_pkgs, 'installed', 'NOT INSTALLED'), '\n'); \
    }"

# Copy application files in one step
COPY app.py likert_pattern_analysis.py likert_analysis.R likert_hybrid.py utils.py ./
COPY templates/ ./templates/

# Expose port and start command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

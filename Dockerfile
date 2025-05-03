# 1. Use Rocker’s R image for reproducibility and binary packages
FROM rocker/r-ver:4.3.2

# 2. Install system dependencies with extended apt timeouts in a single layer
RUN echo 'Acquire::Retries "5"; Acquire::http::Timeout "60"; Acquire::https::Timeout "60";' \
      > /etc/apt/apt.conf.d/99timeouts \
  && apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       gfortran \
       libcurl4-openssl-dev \
       libssl-dev \
       libxml2-dev \
  && rm -rf /var/lib/apt/lists/*

# 3. Configure R to use RStudio Package Manager for pre-compiled binaries
#    and set parallel CPUs and download timeout
RUN echo 'options(repos = c(CRAN="https://packagemanager.posit.co/cran/__linux__/focal/latest"), \
                 Ncpus = 4, timeout = 600)' \
     >> "${R_HOME}/etc/Rprofile.site" :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

# 4. Install 'pak' for fast, parallel R package installs
RUN Rscript -e "install.packages('pak', repos = 'https://r-lib.github.io/p/pak/dev/')" :contentReference[oaicite:6]{index=6}

# 5. Copy and install R packages from cacheable Requirements.R
WORKDIR /app
COPY Requirements.R /app/Requirements.R
RUN Rscript -e "pak::pkg_install(readLines('Requirements.R'))" :contentReference[oaicite:7]{index=7}

# 6. Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt :contentReference[oaicite:8]{index=8}

# 7. Copy application code
COPY . /app

# 8. Expose port and default command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

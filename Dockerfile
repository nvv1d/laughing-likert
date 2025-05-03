# 1. Base image with R
FROM rocker/r-ver:4.3.2

# 2. System dependencies with extended apt timeouts
RUN echo 'Acquire::Retries "5"; Acquire::http::Timeout "60"; Acquire::https::Timeout "60";' \
      > /etc/apt/apt.conf.d/99timeouts \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential \
       gfortran \
       libcurl4-openssl-dev \
       libssl-dev \
       libxml2-dev \
  && rm -rf /var/lib/apt/lists/*

# 3. Configure CRAN mirror, parallel CPUs, and timeout
RUN echo 'options(repos = c(CRAN="https://packagemanager.posit.co/cran/__linux__/focal/latest"), \
                 Ncpus = 4, timeout = 600)' \
     >> "${R_HOME}/etc/Rprofile.site"

# 4. Install pak for fast, parallel R package installs
RUN Rscript -e "install.packages('pak', repos='https://r-lib.github.io/p/pak/dev/')"

# 5. Copy and install R packages via Requirements.R (cacheable layer)
WORKDIR /app
COPY Requirements.R /app/Requirements.R
RUN Rscript -e "pak::pkg_install(readLines('Requirements.R'))"

# 6. Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code
COPY . /app

# 8. Expose port and default command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

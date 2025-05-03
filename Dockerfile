FROM rocker/r-ver:4.4.2

# 1. System deps + apt timeouts
RUN echo 'Acquire::Retries "5"; Acquire::http::Timeout "60"; Acquire::https::Timeout "60";' \
      > /etc/apt/apt.conf.d/99timeouts \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential gfortran libcurl4-openssl-dev libssl-dev libxml2-dev \
  && rm -rf /var/lib/apt/lists/*

# 2. R repo, parallel CPUs, timeout
RUN echo 'options(repos = c(CRAN="https://packagemanager.posit.co/cran/__linux__/focal/latest"), \
                 Ncpus = 4, timeout = 600)' \
     >> "${R_HOME}/etc/Rprofile.site"

# 3. Install pak
RUN Rscript -e "install.packages('pak', repos='https://r-lib.github.io/p/pak/dev/')"

WORKDIR /app

# 4. Copy & install R packages
COPY Requirements.R /app/
RUN Rscript -e "source('Requirements.R'); pak::pkg_install(packages)"

# 5. Python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 6. App code
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

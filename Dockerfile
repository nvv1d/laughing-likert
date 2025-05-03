# 1. Base image: R ≥ 4.4.0
FROM rocker/r-ver:4.4.2

# 2. System deps: R build tools + Python, pip, and venv support
RUN echo 'Acquire::Retries "5"; Acquire::http::Timeout "60"; Acquire::https::Timeout "60";' \
      > /etc/apt/apt.conf.d/99timeouts \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential \
       gfortran \
       libcurl4-openssl-dev \
       libssl-dev \
       libxml2-dev \
       python3-full \
       python3-venv \
       python3-pip \
  && rm -rf /var/lib/apt/lists/*

# 3. Configure CRAN mirror, parallel CPUs, timeout
RUN echo 'options(repos = c(CRAN="https://packagemanager.posit.co/cran/__linux__/focal/latest"), \
                 Ncpus = 4, timeout = 600)' \
     >> "${R_HOME}/etc/Rprofile.site"

# 4. Install pak for parallel R installs
RUN Rscript -e "install.packages('pak', repos='https://r-lib.github.io/p/pak/dev/')"

WORKDIR /app

# 5. Copy & install R packages
COPY Requirements.R /app/
RUN Rscript -e "source('Requirements.R'); pak::pkg_install(packages)"

# 6. Set up Python venv and install Python dependencies
#    - Create venv in /opt/venv
#    - Add venv bin to PATH
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#    - Copy requirements and install into venv
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code
COPY . /app

# 8. Expose port and run Streamlit with venv Python
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

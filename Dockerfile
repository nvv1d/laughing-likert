# Start from standard R image with more built-in libraries
FROM rocker/tidyverse:4.3

# Use all CPU cores for compilation
ENV MAKEFLAGS="-j$(nproc)"

# Ensure R compiles with C++17
RUN mkdir -p /root/.R && \
    echo "CXX11 = g++ -std=gnu++17" >> /root/.R/Makevars && \
    echo "CXX11FLAGS += -O2"        >> /root/.R/Makevars && \
    echo "CXXFLAGS += -O2"          >> /root/.R/Makevars

# Install essential system deps and available CRAN binary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-setuptools python3-wheel \
    build-essential gfortran libgsl-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libblas-dev liblapack-dev libopenblas-dev \
    libbz2-dev liblzma-dev zlib1g-dev \
    libfontconfig1-dev libfribidi-dev libharfbuzz-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libcairo2-dev \
    r-cran-readr r-cran-readxl r-cran-polycor r-cran-psych r-cran-lavaan \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install remaining R packages (including mokken & simstudy) in parallel
RUN Rscript -e "options(Ncpus = parallel::detectCores()); \
    install.packages('remotes', repos = 'https://cloud.r-project.org'); \
    remotes::install_cran(c('readr','readxl','polycor','psych','lavaan','simstudy','mokken','semTools'), \
                          dependencies=TRUE, repos='https://cloud.r-project.org'); \
    remotes::install_cran(c('mirt','ltm','lordif'), \
                          dependencies=TRUE, repos='https://cloud.r-project.org'); \
    pkgs <- c('readr','readxl','polycor','psych','lavaan','simstudy','mokken','semTools','mirt','ltm','lordif'); \
    miss <- setdiff(pkgs, installed.packages()[,'Package']); \
    if(length(miss)) stop('Missing R packages: ', paste(miss, collapse=',')) else cat('All R packages installed ✔\n')"

# Configure Python and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py likert_pattern_analysis.py likert_analysis.R likert_hybrid.py utils.py ./
COPY templates/ ./templates/

# Expose port and start Streamlit
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]

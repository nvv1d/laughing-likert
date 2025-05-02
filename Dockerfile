# Start from standard R image with more built-in libraries
FROM rocker/tidyverse:4.3

# Install essential system dependencies (including Python and compression libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libblas-dev \
    liblapack-dev \
    libgfortran-11-dev \
    libopenblas-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    libfontconfig1-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libgsl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install R packages with a proven approach
RUN R -e "install.packages('remotes', repos = 'https://cloud.r-project.org'); \
    remotes::install_cran(c('readr', 'readxl', 'polycor', 'psych', 'lavaan', 'simstudy', 'mokken'), \
                         dependencies = TRUE, type = 'binary', repos = 'https://cloud.r-project.org'); \
    remotes::install_cran('semTools', dependencies = TRUE, repos = 'https://cloud.r-project.org');"

# Install lordif separately with specific dependencies
RUN R -e "install.packages(c('mirt', 'ltm'), repos = 'https://cloud.r-project.org'); \
    install.packages('lordif', repos = 'https://cloud.r-project.org', \
                   dependencies = TRUE, \
                   INSTALL_opts = c('--no-multiarch'));"

# Verify R packages installation
RUN R -e "required_pkgs <- c('readr', 'readxl', 'polycor', 'psych', 'lavaan', 'simstudy', 'mokken', 'semTools', 'lordif'); \
    installed_pkgs <- installed.packages()[,'Package']; \
    missing <- required_pkgs[!required_pkgs %in% installed_pkgs]; \
    if(length(missing) > 0) { \
        stop(paste('Failed to install:', paste(missing, collapse=', '))); \
    } else { \
        cat('All R packages successfully installed!'); \
    }"

# Configure Python and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Copy requirements first to leverage layer caching
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files in one step
COPY app.py likert_pattern_analysis.py likert_analysis.R likert_hybrid.py utils.py ./
COPY templates/ ./templates/

# Expose port and start command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

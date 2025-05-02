# Use rocker/r-ver as base - already optimized for R
FROM rocker/r-ver:4.3.0

# Install system dependencies for Python and your R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
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
    zlib1g-dev \
    libfontconfig1-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install R package manager and dependencies
RUN R -e "\
    install.packages('pak', repos = 'https://cloud.r-project.org'); \
    pak::pkg_install(c('readr', 'readxl'), ask = FALSE); \
    pak::pkg_install(c('polycor', 'psych'), ask = FALSE); \
    pak::pkg_install(c('lavaan', 'semTools'), ask = FALSE); \
    pak::pkg_install(c('simstudy', 'mokken'), ask = FALSE); \
    pak::pkg_install('lordif', ask = FALSE); \
    installed_pkgs <- installed.packages()[,'Package']; \
    required_pkgs <- c('readr', 'readxl', 'polycor', 'psych', 'lavaan', 'simstudy', 'mokken', 'semTools', 'lordif'); \
    missing <- required_pkgs[!required_pkgs %in% installed_pkgs]; \
    if(length(missing) > 0) { \
      stop(paste('Failed to install:', paste(missing, collapse=', '))); \
    } \
    cat('All packages successfully installed!')"

# Configure Python 
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
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

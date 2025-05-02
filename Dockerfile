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
# Install R packages in separate steps for better error handling
RUN R -e "install.packages('pak', repos = 'https://cloud.r-project.org')"

# Install packages in groups to isolate failures
RUN R -e "pak::pkg_install(c('readr', 'readxl'), ask = FALSE)"
RUN R -e "pak::pkg_install(c('polycor', 'psych'), ask = FALSE)"
RUN R -e "pak::pkg_install(c('lavaan', 'semTools'), ask = FALSE)"
RUN R -e "pak::pkg_install(c('simstudy', 'mokken'), ask = FALSE)"

# Install lordif with extra dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgsl-dev \
    libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use binary packages when possible and install lordif with system library paths
# First attempt with pak
RUN R -e "options(pkgType = 'binary'); \
    options(install.packages.compile.from.source = 'never'); \
    Sys.setenv(CXXFLAGS = '-I/usr/include/eigen3'); \
    Sys.setenv(PKG_CONFIG_PATH = '/usr/lib/x86_64-linux-gnu/pkgconfig'); \
    tryCatch({ \
        pak::pkg_install('lordif', ask = FALSE, dependencies = TRUE) \
    }, error = function(e) { \
        cat('First method failed, trying alternate installation method\n') \
    })"

# Fallback method - direct installation with specific dependency handling
RUN R -e "if (!require('lordif', quietly = TRUE)) { \
        install.packages('mirt', repos = 'https://cloud.r-project.org'); \
        install.packages('ltm', repos = 'https://cloud.r-project.org'); \
        install.packages('https://cran.r-project.org/src/contrib/lordif_0.3-3.tar.gz', \
                        repos = NULL, \
                        type = 'source', \
                        INSTALL_opts = c('--no-multiarch')) \
    }"

# Verify all packages are installed
RUN R -e "required_pkgs <- c('readr', 'readxl', 'polycor', 'psych', 'lavaan', 'simstudy', 'mokken', 'semTools', 'lordif'); \
    installed_pkgs <- installed.packages()[,'Package']; \
    missing <- required_pkgs[!required_pkgs %in% installed_pkgs]; \
    if(length(missing) > 0) { \
        stop(paste('Failed to install:', paste(missing, collapse=', '))); \
    } else { \
        cat('All packages successfully installed!'); \
    }"

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

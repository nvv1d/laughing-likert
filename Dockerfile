# Start from standard R image with more built-in libraries
FROM rocker/tidyverse:4.3

# Use all CPU cores for compilation
ENV MAKEFLAGS="-j$(nproc)"

# (Optional) Ensure R compiles with C++17 - uncomment if needed
# RUN mkdir -p /root/.R && \
#     echo "CXX11 = g++ -std=gnu++17" >> /root/.R/Makevars && \
#     echo "CXX11FLAGS += -O2"        >> /root/.R/Makevars && \
#     echo "CXXFLAGS += -O2"          >> /root/.R/Makevars

# Install essential system deps for R package compilation (including NLopt & CMake for nloptr)
# Removed r-cran-* packages, added libgit2-dev for pak
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-setuptools python3-wheel \
    build-essential gfortran libgsl-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libblas-dev liblapack-dev libopenblas-dev \
    libbz2-dev liblzma-dev zlib1g-dev \
    libfontconfig1-dev libfribidi-dev libharfbuzz-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libcairo2-dev \
    libnlopt-dev cmake pkg-config libgit2-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install pak and then use pak to install R packages (often faster and uses binaries)
# Set Ncpus option for pak as well
RUN Rscript -e "options(Ncpus = parallel::detectCores()); \
    install.packages('pak', repos = sprintf('https://r-lib.github.io/p/pak/stable/%s/%s/%s', .Platform\$pkgType, R.Version()\$os, R.Version()\$arch)); \
    library(pak); \
    cat('Installing R packages using pak...\n'); \
    pkg_install(c('readr','readxl','polycor','psych','lavaan','simstudy','mokken','semTools','nloptr','mirt','ltm','lordif'), \
                dependencies=TRUE); \
    cat('Checking installed packages...\n'); \
    pkgs <- c('readr','readxl','polycor','psych','lavaan','simstudy','mokken','semTools','nloptr','mirt','ltm','lordif'); \
    miss <- setdiff(pkgs, installed.packages()[,'Package']); \
    if(length(miss)) stop(paste('Missing R packages after pak install:', paste(miss, collapse=', '))) else cat('All R packages installed via pak ✔\n')"

# Configure Python and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py likert_pattern_analysis.py likert_analysis.R likert_hybrid.py utils.py ./
COPY templates/ ./templates/

# Expose the port Railway expects the app to listen on (value provided by $PORT env var)
EXPOSE 8501 # This is informational, Railway uses $PORT at runtime

# Start Streamlit using the $PORT environment variable provided by Railway
CMD ["streamlit","run","app.py","--server.port=$PORT","--server.address=0.0.0.0"]

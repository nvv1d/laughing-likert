# 1. Base image
FROM python:3.11-slim@sha256:75a17dd6f00b277975715fc094c4a1570d512708de6bb4c5dc130814813ebfe4

# 2. Install system deps and R base
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      libblas-dev \
      liblapack-dev \
      libcurl4-openssl-dev \
      libssl-dev \
      libxml2-dev \
      ca-certificates \
      gnupg \
      software-properties-common \
 && rm -rf /var/lib/apt/lists/*

# 3. Add R 4.2 repository from CRAN
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E19F5F87128899B192B1A2C2AD5F960A256A04AF \
 && add-apt-repository "deb http://cloud.r-project.org/bin/linux/debian bookworm-cran42/" \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      r-base \
      r-base-dev \
      r-recommended \
      r-cran-tidyverse \
      r-cran-polycor \
      r-cran-psych \
      r-cran-lavaan \
 && rm -rf /var/lib/apt/lists/*

# 4. Working directory
WORKDIR /app

# 5. Configure R
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)' > /etc/R/Rprofile.site \
 && mkdir -p /root/.R \
 && echo 'MAKEFLAGS = -j4' > /root/.R/Makevars

# 6. Install remaining R packages with increased timeout
# Install each package separately to identify issues
RUN Rscript -e "if(!require('simstudy')) { install.packages('simstudy', dependencies=TRUE) }"
RUN Rscript -e "if(!require('mokken')) { install.packages('mokken', dependencies=TRUE) }"
RUN Rscript -e "if(!require('semTools')) { install.packages('semTools', dependencies=TRUE) }"
RUN Rscript -e "if(!require('lordif')) { install.packages('lordif', dependencies=TRUE) }"

# 7. Verify R & key packages
RUN Rscript -e "lapply(c('readr', 'readxl', 'polycor', 'psych', 'simstudy', 'mokken', 'lavaan', 'semTools', 'lordif'), function(pkg) { cat(paste('Loading:', pkg, '\n')); library(pkg, character.only=TRUE) }); sessionInfo()"

# 8. Copy & install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 9. Copy application code
COPY app.py .
COPY likert_pattern_analysis.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY utils.py .

# 10. Copy templates
COPY templates/ ./templates/

# 11. Expose port & default command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

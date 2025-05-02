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
      r-base \
      r-base-dev \
      r-cran-readr \
      r-cran-readxl \
      r-cran-polycor \
      r-cran-psych \
      r-cran-lavaan \
      # For faster downloads
      ca-certificates \
      apt-transport-https \
      libfontconfig1-dev \
 && rm -rf /var/lib/apt/lists/*

# Set CRAN mirror in Rprofile for reliability
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' > /etc/R/Rprofile.site

# 3. Working directory
WORKDIR /app

# 4. Install R packages from CRAN (in smaller batches with timeout)
RUN Rscript -e "options(timeout=600); install.packages('simstudy', repos='https://cloud.r-project.org/', dependencies=TRUE, verbose=TRUE)"
RUN Rscript -e "options(timeout=600); install.packages('mokken', repos='https://cloud.r-project.org/', dependencies=TRUE, verbose=TRUE)"
RUN Rscript -e "options(timeout=600); install.packages('semTools', repos='https://cloud.r-project.org/', dependencies=TRUE, verbose=TRUE)"
RUN Rscript -e "options(timeout=600); install.packages('lordif', repos='https://cloud.r-project.org/', dependencies=TRUE, verbose=TRUE)"

# 5. Verify R & key packages (with more verbose output)
RUN Rscript -e "for(pkg in c('readr', 'readxl', 'polycor', 'psych', 'simstudy', 'mokken', 'lavaan', 'semTools', 'lordif')) { cat(paste('Loading package:', pkg, '\n')); library(pkg, character.only=TRUE) }; sessionInfo()"

# 6. Copy & install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code
COPY app.py .
COPY likert_pattern_analysis.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY utils.py .

# 8. Copy templates
COPY templates/ ./templates/

# 9. Expose port & default command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

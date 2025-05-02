# 1. First stage - build R environment
FROM r-base:4.2 AS r-base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure R to use parallel compilation
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4, timeout = 600)' > /etc/R/Rprofile.site

# Install required R packages
RUN R -e "install.packages(c('readr', 'readxl', 'polycor', 'psych', 'lavaan'), dependencies = TRUE)" && \
    R -e "install.packages('simstudy', dependencies = TRUE)" && \
    R -e "install.packages('mokken', dependencies = TRUE)" && \
    R -e "install.packages('semTools', dependencies = TRUE)" && \
    R -e "install.packages('lordif', dependencies = TRUE)"

# 2. Second stage - Python with R
FROM python:3.11-slim

# Copy R from first stage
COPY --from=r-base /usr/local/lib/R /usr/local/lib/R
COPY --from=r-base /usr/local/bin/R /usr/local/bin/R
COPY --from=r-base /usr/local/bin/Rscript /usr/local/bin/Rscript
COPY --from=r-base /etc/R /etc/R

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    libssl3 \
    libxml2 \
    libgomp1 \
    libfontconfig1 \
    libharfbuzz0b \
    libfribidi0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working directory
WORKDIR /app

# 4. Verify R & key packages
RUN Rscript -e "lapply(c('readr', 'readxl', 'polycor', 'psych', 'simstudy', 'mokken', 'lavaan', 'semTools', 'lordif'), function(pkg) { cat(paste('Loading:', pkg, '\n')); suppressPackageStartupMessages(library(pkg, character.only=TRUE)) }); sessionInfo()"

# 5. Copy & install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY app.py .
COPY likert_pattern_analysis.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY utils.py .

# 7. Copy templates
COPY templates/ ./templates/

# 8. Expose port & default command
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

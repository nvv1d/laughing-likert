FROM python:3.11-slim

WORKDIR /app

# Install required system dependencies for R
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install necessary R packages
RUN R -e "install.packages(c('psych', 'lavaan'), repos='https://cloud.r-project.org/')"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p ./data ./templates

# Copy application files
COPY app.py .
COPY utils.py .
COPY likert_analysis.R .
COPY likert_hybrid.py .
COPY likert_pattern_analysis.py .
COPY templates/ ./templates/
COPY data/ ./data/

# Create Streamlit config directory and add configuration
RUN mkdir -p ./.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 5000\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
' > ./.streamlit/config.toml

# Expose the application port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
FROM python:3.11-slim

# Setup app as working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY app.py .
COPY custom_transformers.py .
COPY constants.py .

# Expose port for Streamlit (default Streamlit port)
EXPOSE 8501

# Command to run the application
# The --server.address=0.0.0.0 flag is important as it allows Streamlit to be accessible outside the container.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

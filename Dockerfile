FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Streamlit-specific commands; allow Streamlit to run on port 8501 in the container
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

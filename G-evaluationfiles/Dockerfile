# Use an official Python runtime as a parent image
FROM python:3.12-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 (the default Streamlit port)
EXPOSE 8501

# Expose port 8502 (for the second Streamlit app, if needed)
EXPOSE 8502

# Start both Streamlit apps
CMD ["sh", "-c", "streamlit run file1.py & streamlit run file2.py --server.port=8502"]

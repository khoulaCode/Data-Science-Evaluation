# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container (for Streamlit)
EXPOSE 8501

# Define environment variable for Streamlit to run on the specified host
ENV STREAMLIT_SERVER_HEADLESS="true"
ENV STREAMLIT_SERVER_PORT="8501"
ENV STREAMLIT_SERVER_ENABLECORS="false"
ENV STREAMLIT_SERVER_ENABLEWEBBROWSER="false"

# Run your Python script when the container launches
CMD ["streamlit", "run", "ML_Evaluation.py"]

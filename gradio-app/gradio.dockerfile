# Use an official Python runtime as a parent image
FROM python:3.12.9-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app
COPY ./requirements.txt /app

# Install pip requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install other libraries
RUN pip install --no-cache-dir gradio

# Define environment variable

# Make port available to the world outside this container
EXPOSE 7860

# Run Jupyter Notebook when the container launches
CMD ["python", "app.py", "--listen", "--server-port", "7860", "--server-name", "0.0.0.0"]
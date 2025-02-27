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
RUN pip install --no-cache-dir jupyterlab

# Define environment variable

# Make port available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
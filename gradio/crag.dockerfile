# Use an official Python runtime as a parent image
FROM langchain/langchain:0.1.0

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install other libraries
RUN pip install --no-cache-dir gradio

# Define environment variable

# Copy the current directory contents into the container at /app
#COPY . /app

# Make port available to the world outside this container
EXPOSE 7860

# Run Jupyter Notebook when the container launches
CMD ["python", "crag.py", "--listen", "--server-port", "7860", "--server-name", "0.0.0.0"]
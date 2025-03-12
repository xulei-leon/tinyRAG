# Use an official Python runtime as a parent image
FROM python:3.12.9-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install other libraries
RUN pip install --no-cache-dir gradio
RUN pip install --no-cache-dir langchain
RUN pip install --no-cache-dir langchain-core
RUN pip install --no-cache-dir langchain-community
RUN pip install --no-cache-dir langchain-openai
RUN pip install --no-cache-dir langchain-deepseek
RUN pip install --no-cache-dir langchain-text-splitters 
RUN pip install --no-cache-dir langchain-chroma
RUN pip install --no-cache-dir langchain-huggingface
RUN pip install --no-cache-dir langchain-linkup
RUN pip install --no-cache-dir langgraph
RUN pip install --no-cache-dir langgraph-sdk
RUN pip install --no-cache-dir langgraph-checkpoint-sqlite
RUN pip install --no-cache-dir langgraph-cli
RUN pip install --no-cache-dir langsmith
RUN pip install --no-cache-dir tavily-python
RUN pip install --no-cache-dir wikipedia
RUN pip install --no-cache-dir trustcall
RUN pip install --no-cache-dir chromadb
RUN pip install --no-cache-dir requests
RUN pip install --no-cache-dir beautifulsoup4
RUN pip install --no-cache-dir pypdf
RUN pip install --no-cache-dir python-magic
RUN pip install --no-cache-dir unstructured

# Define environment variable

# Copy the current directory contents into the container at /app
#COPY . /app

# Make port available to the world outside this container
EXPOSE 7860

# Run Jupyter Notebook when the container launches
CMD ["python", "web-app.py", "--listen", "--server-port", "7860", "--server-name", "0.0.0.0"]
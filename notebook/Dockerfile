# Use an official Python runtime as a parent image
FROM python:3.12.9-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install pip requirements
RUN pip install --no-cache-dir langchain
RUN pip install --no-cache-dir langchain-core
RUN pip install --no-cache-dir langchain-community
RUN pip install --no-cache-dir langchain-openai
RUN pip install --no-cache-dir langchain-deepseek
RUN pip install --no-cache-dir langchain-text-splitters 
RUN pip install --no-cache-dir langchain-chroma
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
RUN pip install --no-cache-dir langchain-huggingface
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir llama-index
RUN pip install --no-cache-dir llama-index-core
RUN pip install --no-cache-dir llama-index-embeddings-huggingface
RUN pip install --no-cache-dir llama-index-vector-stores-chroma
RUN pip install --no-cache-dir llama-index-llms-langchain

# Install other libraries
RUN pip install --no-cache-dir jupyterlab
RUN pip install --no-cache-dir ipywidgets

# Define environment variable

# Make port available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]

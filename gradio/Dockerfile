FROM python:3.12.9-slim-bookworm AS builder

# Predownload the model to the /models directory
WORKDIR /models
RUN pip install huggingface-hub
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='BAAI/bge-small-zh-v1.5', \
    local_dir='BAAI/bge-small-zh-v1.5')"
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='BAAI/bge-reranker-base', \
    local_dir='BAAI/bge-reranker-base')"

# Use an official Python runtime as a parent image
FROM python:3.12.9-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install libraries
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
RUN pip install --no-cache-dir requests
RUN pip install --no-cache-dir beautifulsoup4
RUN pip install --no-cache-dir pypdf
RUN pip install --no-cache-dir python-magic
RUN pip install --no-cache-dir unstructured
RUN pip install --no-cache-dir rank_bm25

# Install other libraries
RUN pip install --no-cache-dir gradio

# Install Hugging Face model
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers --no-deps
RUN pip install --no-cache-dir transformers tqdm scikit-learn numpy
RUN pip install --no-cache-dir langchain-huggingface --no-deps

COPY ./debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt-get update
RUN apt-get install -y libmagic1

RUN pip install --no-cache-dir pickle-mixin
RUN pip install "unstructured[pdf, doc, docx, ppt, pptx]"
RUN pip install "langchain-unstructured[all-docs]"
RUN apt-get install -y libreoffice

# Copy nltk data
ENV NLTK_DATA=/nltk_data
COPY ./nltk_data ${NLTK_DATA}

# Download nltk data
###############################################################################
# RUN pip install nltk
# ARG NLTK_DATASETS="punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords"
# RUN mkdir -p ${NLTK_DATA}
# RUN for dataset in ${NLTK_DATASETS}; do \
#     python -c "import nltk; nltk.download('${dataset}', download_dir='${NLTK_DATA}')"; \
# done
###############################################################################

# Define environment variable
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy the models from the previous image
COPY --from=builder /models /models

# Copy the current directory contents into the container at /app
#COPY . /app

# Make port available to the world outside this container
EXPOSE 7860

# Run bash for debug
#CMD ["bash"]

# Run app when the container launches
CMD ["python", "web-app.py", "--listen", "--server-port", "7860", "--server-name", "0.0.0.0"]
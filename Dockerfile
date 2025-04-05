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

# Install system dependencies
COPY config/debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt-get update
RUN apt-get install -y libmagic1

# Install python libraries
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face model
RUN pip install --no-cache-dir torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu & \
    pip install --no-cache-dir sentence-transformers --no-deps & \
    pip install --no-cache-dir transformers tqdm scikit-learn numpy & \
    pip install --no-cache-dir langchain-huggingface --no-deps & \
    pip install torchvision --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir pickle-RUN pip install --no-cache-dir mixin & \
    pip install --no-cache-dir "unstructured[text, csv, markdown, json]" & \
    pip install --no-cache-dir "langchain-unstructured[all-docs]"


# Copy nltk data
ENV NLTK_DATA=/nltk_data
COPY nltk_data ${NLTK_DATA}

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
# CMD ["bash"]

# Run app when the container launches
CMD ["python", "src/app.py", "--listen", "--server-port", "7860", "--server-name", "0.0.0.0"]

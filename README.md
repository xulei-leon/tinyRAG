# AI Agent Learning
Some sample code to help you learn about LangChain and LangGraph for building AI agents.

## Download nltk data
Windows VScode TERMINAL
```bash
> mkdir nltk_data

> pip install nltk
> python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords'], download_dir='nltk_data')"
```

## Build docker image
```bash
> cp key.env .env
<Add your token and key>

> docker compose build
```

## Run Shell to prepare retrieve index database
```bash
> docker run  -v .:/app/ -w /app/ --name shell -it tinyrag:latest bash
```

### Execute the command in your docker image shell
Install libraries for loading pdf, doc, docx, ppt, pptx documents
```bash
apt-get install -y libreoffice
pip install torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install "unstructured[pdf, doc, docx, ppt, pptx]"
```

Create bm25 index from your documents files
```bash
$ python src/rag_retriever.py --bm25 /app/downloads/files
```

Create vector index from your documents files
```bash
$ python src/rag_retriever.py --vector /app/downloads/files
```


## Run docker compose image
```bash
> docker compose up -d
```

### Run gradio CRAG app

Open web brower and input: http://127.0.0.1/8080
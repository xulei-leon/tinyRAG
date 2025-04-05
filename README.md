# AI Agent Learning
Some sample code to help you learn about LangChain and LangGraph for building AI agents.

### Download nltk data
Windows VScode TERMINAL
```bash
> mkdir nltk_data

> pip install nltk
> python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords'], download_dir='nltk_data')"
```

### Build docker image
```bash
$ docker compose build
```

### Run Shell to prepare retrieve index database
```bash
docker run  -v .:/app/ -w /app/ --name shell -it tinyrag:latest bash
```

### Run docker compose image

```bash
$ docker compose up -d

$ docker compose ps
NAME                IMAGE             COMMAND                   SERVICE    CREATED         STATUS          PORTS
gradio-crag-app-1   gradio-app:latest   "python web-app.py -…"   crag-app   50 seconds ago   Up 47 seconds   0.0.0.0:7860->7860/tcp
```

### Run gradio CRAG app

Open web brower and input: http://127.0.0.1/7
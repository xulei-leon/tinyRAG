# AI Agent Learning
Some sample code to help you learn about LangChain and LangGraph for building AI agents.

## Jupyter Notebook
### Build docker

```bash
$ cd notebook
$ docker build -t langchain:latest .
```

### Run docker

```bash
docker run -d -p 8888:8888 -v .:/app/notebook -w /app/notebook --name langchain-notebook langchain:0.3
```

### Run Jupyter Notebook with your web brower.

Open web brower and input: http://127.0.0.1:8888/lab/tree/notebook

## Gradio web app
### Build docker compose image

```bash
$ cd gradio
$ docker compose build

$ docker image ls
REPOSITORY         TAG                    IMAGE ID       CREATED              SIZE
gradio-app         latest                 a42822ef3991   5 minutes ago   6.87GB
```

### Run docker compose image

```bash
$ docker compose up -d

$ docker compose ps
NAME                IMAGE             COMMAND                   SERVICE    CREATED         STATUS          PORTS
gradio-crag-app-1   gradio-app:latest   "python web-app.py -…"   crag-app   50 seconds ago   Up 47 seconds   0.0.0.0:7860->7860/tcp

$ docker compose restart crag-app
```

### Rradio CRAG app

Open web brower and input: http://127.0.0.1/7860

### Debug login as bash

```bash
$ docker exec -it gradio-crag-app-1 bash
```

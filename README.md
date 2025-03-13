# AI Agent Learning
Some sample code to help you learn about LangChain and LangGraph for building AI agents.

## Jupyter Notebook
### Build docker

<pre><code class="shell">
docker build -t langchain:0.3 -f ./notebook/jupyter.dockerfile .
</code></pre>

### Run docker

<pre><code class="shell">
docker run -d -p 8888:8888 -v .:/app/notebook -w /app/notebook --name langchain-notebook langchain:0.3
</code></pre>

### Run Jupyter Notebook with your web brower.

Open web brower and input: http://127.0.0.1:8888/lab/tree/notebook

## Gradio web app
### Build docker compose image

<pre><code class="shell">
$ cd gradio
$ docker compose build

$ docker image ls
REPOSITORY         TAG                    IMAGE ID       CREATED              SIZE
gradio-app         latest                 a42822ef3991   5 minutes ago   6.87GB
</code></pre>

### Run docker compose image

<pre><code class="shell">
$ docker compose up -d

$ docker compose ps
NAME                IMAGE             COMMAND                   SERVICE    CREATED         STATUS          PORTS
gradio-crag-app-1   gradio-app:latest   "python web-app.py -…"   crag-app   50 seconds ago   Up 47 seconds   0.0.0.0:7860->7860/tcp

$ docker compose restart crag-app
</code></pre>

### Rradio CRAG app

Open web brower and input: http://127.0.0.1/7860

### Debug login as bash

<pre><code class="shell">
$ docker exec -it gradio-crag-app-1 bash
</code></pre>

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
$ docker compose -f gradio/crag-docker-compose.yml build

$ docker image ls
REPOSITORY            TAG                    IMAGE ID       CREATED         SIZE
crag-app              latest                 4c2c91cfc18d   2 hours ago     1.58GB
</code></pre>

### Run docker compose image

<pre><code class="shell">
$ docker compose -f gradio/crag-docker-compose.yml up -d

$ docker compose -f gradio/crag-docker-compose.yml ps     
NAME                IMAGE             COMMAND                   SERVICE    CREATED         STATUS          PORTS
gradio-crag-app-1   crag-app:latest   "python crag.py --li…"   crag-app   3 minutes ago   Up 17 seconds   0.0.0.0:7860->7860/tcp
</code></pre>
 

### Rradio CRAG app

Open web brower and input: http://127.0.0.1/7860


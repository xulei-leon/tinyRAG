import os
import sys
import tomllib
import gradio as gr
from dotenv import load_dotenv
import tomllib

# langchain
from langchain_deepseek import ChatDeepSeek
from langchain_community.retrievers import TavilySearchAPIRetriever

# my modules
from rag_graph import RagGraph
from rag_retriever import RagRetriever
from llm_processor import LLMProcessor


def stream_response(inputs, config):
    for output in rag_app.stream(inputs, config):
        for node_name, node_state in output.items():
            if not node_state:
                continue

            if node_state.get("thinking"):
                yield ">" + node_state["thinking"]

            if node_state.get("answer"):
                yield node_state["answer"]


# Define a function to run the conversation
def run_conversation(user_input, chat_history):
    thread_id = "gradio_test"
    chat_history.append((user_input, ""))
    full_response = []

    inputs = {"question": user_input}
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in stream_response(inputs, config):
        full_response.append(chunk)
        chat_history[-1] = (user_input, "\n\n".join(full_response))
        yield "", chat_history

    chat_history[-1] = (user_input, chat_history[-1][1])
    yield "", chat_history


with open("config/config.toml", "rb") as f:
    config_data = tomllib.load(f)
    embed_model = config_data.get("huggingface", {}).get("embed_model")
    reranker_model = config_data.get("huggingface", {}).get("reranker_model")
    files_directory = config_data.get("vector", {}).get("files_directory")
    persist_directory = config_data.get("vector", {}).get("persist_directory")
    collection_name = config_data.get("vector", {}).get("collection_name")
    deepseek_llm_model = config_data.get("deepseek", {}).get("model")
    deepseek_llm_temperature = config_data.get("deepseek", {}).get("temperature")
    deepseek_llm_max_tokens = config_data.get("deepseek", {}).get("max_tokens")
    web_search_num = config_data.get("retriever", {}).get("web_search_num", 1)
    chat_agent_name = config_data.get("chat", {}).get("chat_agent_name")

# Create a Gradio interface
with gr.Blocks() as agent:
    gr.Markdown("#" + chat_agent_name)
    chatbot = gr.Chatbot(type="tuples")

    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(run_conversation, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
linkup_api_key = os.getenv("LINKUP_API_KEY")


print("=== Init Config ===")
print(f"model_name: {embed_model}")
print(f"reranker_model: {reranker_model}")
print(f"files_directory: {files_directory}")
print(f"persist_directory: {persist_directory}")
print(f"collection_name: {collection_name}")
print("===================\n")

if not persist_directory:
    print("Error: persist_directory is not defined")
    sys.exit(1)

if not files_directory:
    print("Error: files_directory is not defined")
    sys.exit(1)

if not embed_model:
    print("Error: model_name is not defined")
    sys.exit(1)

llm = ChatDeepSeek(
    api_key=deepseek_api_key,
    model=deepseek_llm_model,
    temperature=deepseek_llm_temperature,
    max_tokens=deepseek_llm_max_tokens,
    timeout=None,
    top_p=0.9,
    frequency_penalty=0.7,
    presence_penalty=0.5,
    max_retries=3,
    streaming=True,
)

llm_processor = LLMProcessor(llm=llm)
rag_retriever = RagRetriever(
    embed_model=embed_model,
    reranker_model=reranker_model,
    persist_directory=persist_directory,
    collection_name=collection_name,
)
rag_retriever.load_index()
web_retriever = TavilySearchAPIRetriever(api_key=tavily_api_key, k=web_search_num)

# Create graph
rag_graph = RagGraph(
    llm_processor=llm_processor,
    rag_retriever=rag_retriever,
    web_retriever=web_retriever,
)
rag_app = rag_graph.compile()

# Set the port for the Gradio app
# Google Cloud Run uses the PORT environment variable
# https://cloud.google.com/run/docs/configuring/environment-variables
port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 8080)))
server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
listen = os.environ.get("GRADIO_LISTEN", "true").lower() == "true"

# Launch the interface
if __name__ == "__main__":
    if listen:
        # agent.launch(pwa=True, share=True)
        agent.launch(server_name=server_name, server_port=port)
    else:
        print("Gradio server is configured not to listen.")

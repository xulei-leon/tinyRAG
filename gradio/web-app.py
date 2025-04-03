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
from crag_graph import CragGraph
from rag_retriever import RagRetriever
from llm_processor import LLMProcessor


def stream_response(inputs):
    for output in rag_app.stream(inputs):
        for node_name, node_state in output.items():
            if node_state.get("thinking"):
                yield node_state["thinking"]

            if node_state.get("answer"):
                yield node_state["answer"]


# Define a function to run the conversation
def run_conversation(user_input, chat_history):
    chat_history.append((user_input, ""))

    inputs = {"question": user_input}
    full_response = []
    for chunk in stream_response(inputs):
        full_response.append(chunk)
        chat_history[-1] = (user_input, "\n\n".join(full_response))
        yield "", chat_history

    chat_history[-1] = (user_input, chat_history[-1][1])
    yield "", chat_history


# Create a Gradio interface
with gr.Blocks() as agent:
    gr.Markdown("# LangGraph CRAG agent")
    chatbot = gr.Chatbot(type="tuples")

    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(run_conversation, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the interface
if __name__ == "__main__":
    load_dotenv()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    silicon_api_key = os.getenv("SILICON_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    linkup_api_key = os.getenv("LINKUP_API_KEY")

    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
        embed_model = config_data.get("huggingface", {}).get("embed_model")
        reranker_model = config_data.get("huggingface", {}).get("reranker_model")
        files_directory = config_data.get("vector", {}).get("files_directory")
        persist_directory = config_data.get("vector", {}).get("persist_directory")
        collection_name = config_data.get("vector", {}).get("collection_name")
        deepseek_llm_model = config_data.get("deepseek", {}).get("model")
        deepseek_llm_temperature = config_data.get("deepseek", {}).get("temperature")
        deepseek_llm_max_tokens = config_data.get("deepseek", {}).get("max_tokens")

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
    web_retriever = TavilySearchAPIRetriever(api_key=tavily_api_key, k=1)

    # Create graph
    crag_graph = CragGraph(
        llm_processor=llm_processor,
        rag_retriever=rag_retriever,
        web_retriever=web_retriever,
    )
    rag_app = crag_graph.compile()

    # agent.launch(pwa=True, share=True)
    agent.launch()

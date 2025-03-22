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
from crag_retriever import CragRetriever
from llm_processor import LLMProcessor


# Define a function to run the conversation
def run_conversation(user_input, chat_history):
    inputs = {"question": user_input}
    response = rag_app.invoke(inputs)

    ai_response = response["answer"]

    chat_history.append((user_input, ai_response))

    return "", chat_history


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
        model_name = config_data.get("huggingface", {}).get("embed_model")
        files_directory = config_data.get("vector", {}).get("files_directory")
        persist_directory = config_data.get("vector", {}).get("persist_directory")
        deepseek_llm_model = config_data.get("deepseek", {}).get("model")
        deepseek_llm_temperature = config_data.get("deepseek", {}).get("temperature")
        deepseek_llm_max_tokens = config_data.get("deepseek", {}).get("max_tokens")

    print(f"model_name: {model_name}")
    print(f"files_directory: {files_directory}")
    print(f"persist_directory: {persist_directory}")

    if not persist_directory:
        print("Error: persist_directory is not defined")
        sys.exit(1)

    if not files_directory:
        print("Error: files_directory is not defined")
        sys.exit(1)

    if not model_name:
        print("Error: model_name is not defined")
        sys.exit(1)

    llm = ChatDeepSeek(
        model=deepseek_llm_model,
        temperature=deepseek_llm_temperature,
        max_tokens=deepseek_llm_max_tokens,
        timeout=None,
        top_p=0.9,
        frequency_penalty=0.7,
        presence_penalty=0.5,
        max_retries=3,
        api_key=deepseek_api_key,
    )

    llm_processor = LLMProcessor(llm=llm)
    rag_retriever = CragRetriever(
        model_name=model_name,
        persist_directory=persist_directory,
    )
    web_retriever = TavilySearchAPIRetriever(api_key=tavily_api_key, k=3)

    # Create graph
    rag_graph = CragGraph(
        llm_processor=llm_processor,
        rag_retriever=rag_retriever,
        web_retriever=web_retriever,
    )
    rag_app = rag_graph.compile()

    # agent.launch(pwa=True, share=True)
    agent.launch()


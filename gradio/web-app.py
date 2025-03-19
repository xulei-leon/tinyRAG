import sys
import tomllib
import gradio as gr
from crag_graph import CragGraph
from crag_retriever import CragRetriever

# Define a function to run the conversation


def run_conversation(user_input, chat_history):
    inputs = {"question": user_input}
    response = rag_app.invoke(inputs)

    ai_response = response["answer"]

    chat_history.append((user_input, ai_response))
    return "", chat_history


# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Chat Demo")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(run_conversation, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the interface
if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
        model_name = config_data.get("huggingface", {}).get("embed_model")
        files_directory = config_data.get("vector", {}).get("files_directory")
        persist_directory = config_data.get("vector", {}).get("persist_directory")

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

    retriever = CragRetriever(
        model_name=model_name,
        persist_directory=persist_directory,
    )

    rag_app = CragGraph(retriever=retriever).compile()

    demo.launch()

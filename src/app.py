import os
import sys
import tomllib
import gradio as gr
from dotenv import load_dotenv
import tomllib

# my modules
from agent import Agent

with open("config/config.toml", "rb") as f:
    config_data = tomllib.load(f)
    if not config_data:
        raise ValueError("Error: config.toml file is empty")

chat_agent_name = config_data.get("chat", {}).get("chat_agent_name")
rag_app = Agent().get_app()


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


# Create a Gradio interface
with gr.Blocks() as agent:
    # gr.Markdown(f"# {chat_agent_name}")
    gr.Markdown(
        f"""
        <div style="text-align: center;">
            <h1>{chat_agent_name}</h1>
        </div>
        """
    )
    chatbot = gr.Chatbot(type="tuples")

    msg = gr.Textbox()
    send = gr.Button("发送问题")
    # clear = gr.Button("清除对话")

    msg.submit(run_conversation, [msg, chatbot], [msg, chatbot])
    # clear.click(lambda: None, None, chatbot, queue=False)
    send.click(run_conversation, [msg, chatbot], [msg, chatbot])


# Set the port for the Gradio app
# Google Cloud Run uses the PORT environment variable
# https://cloud.google.com/run/docs/configuring/environment-variables
port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 8080)))
server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
listen = os.environ.get("GRADIO_LISTEN", "true").lower() == "true"

# Launch the interface
if __name__ == "__main__":
    if listen:
        # agent.launch(share=True, server_name=server_name, server_port=port)
        agent.launch(server_name=server_name, server_port=port)
    else:
        print("Gradio server is configured not to listen.")

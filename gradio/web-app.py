import gradio as gr
from crag_graph import rag_app

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

    msg.submit(
        run_conversation,
        [msg, chatbot],
        [msg, chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the interface
if __name__ == "__main__":
    demo.launch()

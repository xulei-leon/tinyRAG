import gradio as gr

def greet(name):
    # import pdb; pdb.set_trace() # Uncomment to set a pdb breakpoint
    return "Hello " + name + "!"

iface = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Gradio Docker Debug Example App",
    description="A simple Gradio app for demonstrating Docker debugging.",
)

iface.launch(server_name="0.0.0.0", server_port=7860)
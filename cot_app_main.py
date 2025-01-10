import gradio as gr
from cot_reflection_app import iface as iface1 # Assuming block1 is the Gradio block in file1
from cot_reflection_file_app import (
    iface as iface2,
)  

with gr.Blocks() as main_ui:
    with gr.Tabs():
        with gr.TabItem("CoT Basic"):
            iface1.render()  # Render the block from file1 inside its own tab
        with gr.TabItem("CoT with File"):
            iface2.render()  # Render the block from file2 inside its own tab

if __name__ == "__main__":
    # Launch the app
    main_ui.launch(share=False)
import gradio as gr
import os
from cot_reflection_v1 import cot_reflection

def process_question(question, full_response):
    system_prompt = """You are a legal assistant. Provide a detailed and accurate answer to the following question."""
    result = cot_reflection(system_prompt=system_prompt, question=question, return_full_response=full_response)
    return result

# Get the absolute path to the logo file
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "images", "Linklaters.svg.png")

with gr.Blocks() as iface:
    with gr.Column(scale=1):
        gr.Image(logo_path, show_label=False, height=70)
    
    # Add empty space
    gr.Markdown("<br><br>")
    
    gr.Markdown("<h1 style='text-align: center;'>MVP for Chain of Thought Reflection Assistant</h1>")
    
    with gr.Row():
        with gr.Column():
            with gr.Column(scale=1, min_width=600):
                question = gr.Textbox(
                    lines=2, 
                    label="",  # Set an empty label
                    placeholder="Ask a question and get a detailed answer using Chain of Thought reflection powered by Linklaters GenAI Platform."
                )
            full_response = gr.Checkbox(label="Show full response (including chain-of-thought)")
            submit_btn = gr.Button("Submit")
    output = gr.Markdown()
    
    submit_btn.click(
        fn=process_question,
        inputs=[question, full_response],
        outputs=output
    )

if __name__ == "__main__":
    iface.launch(share=False)
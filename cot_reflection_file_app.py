import gradio as gr
import os
import re
from cot_reflection_file import (
    cot_reflection, 
    cot_prompt as default_cot_prompt, 
    system_prompt as default_system_prompt,
    get_model_response,
    AVAILABLE_MODELS
)
from document_utils import read_document

def process_question(file, user_prompt, system_prompt, cot_prompt, selected_model):
    try:
        # Read document content if file is provided
        document_content = None
        if file is not None:
            document_content = read_document(file.name)

        # Get thinking, reflection, and output from cot_reflection
        thinking, reflection, output = cot_reflection(
            system_prompt=system_prompt,
            cot_prompt=cot_prompt,
            question=user_prompt,
            document_content=document_content,
            model_name=selected_model
        )

        # Extract the actual thinking content
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking, re.DOTALL)
        actual_thinking = thinking_match.group(1).strip() if thinking_match else thinking

        # Get the initial response
        doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
        initial_response_prompt = f"{system_prompt}\n\n{doc_content}Question: {user_prompt}\n\nProvide a concise answer to this question without any explanation or reasoning."
        initial_response = get_model_response(selected_model, initial_response_prompt)

        # Provide default messages for empty sections
        initial_response = initial_response if initial_response else "No initial response provided."
        actual_thinking = actual_thinking if actual_thinking else "No thinking process provided."
        reflection = reflection if reflection else "No reflection process provided."
        output = output if output else "No final output provided."

        return user_prompt, initial_response, actual_thinking, reflection, output, system_prompt, cot_prompt
    except Exception as e:
        return user_prompt, f"An error occurred: {str(e)}", "", "", "", system_prompt, cot_prompt

# Get the absolute path to the logo file
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "images", "Linklaters.svg.png")

# Gradio interface
with gr.Blocks() as iface:
    with gr.Column(scale=1):
        gr.Image(logo_path, show_label=False, height=70)
    
    gr.Markdown("<br><br>")
    
    gr.Markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>Document Analysis with Chain of Thought Reflection</h1>
    <p style='text-align: center; font-style: italic; margin-top: 5px; font-size: 1.2em;'>powered by Linklaters GenAI Platform</p>
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Column(scale=1, min_width=600):
                model_selector = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="Gemini 1.5 Pro",
                    label="Select Model",
                    interactive=True
                )
                
                file_input = gr.File(
                    label="Upload Document (DOCX or PDF)",
                    file_types=[".docx", ".pdf"]
                )
                user_prompt = gr.Textbox(
                    lines=2,
                    label="",
                    placeholder="Ask a question about the uploaded document using Chain of Thought reflection."
                )
                with gr.Accordion("Advanced Settings", open=False):
                    system_prompt = gr.Textbox(
                        lines=2,
                        label="System Prompt",
                        value=default_system_prompt
                    )
                    cot_prompt = gr.Textbox(
                        lines=4,
                        label="Chain of Thought Prompt",
                        value=default_cot_prompt
                    )
            submit_btn = gr.Button("Submit")
    
    with gr.Row():
        user_prompt_output = gr.Textbox(label="1. User Prompt")
        initial_response_output = gr.Textbox(label="2. Initial Response")
        thinking_output = gr.Textbox(label="3. Thinking")
        reflection_output = gr.Textbox(label="4. Reflection")
        final_output = gr.Textbox(label="5. Final Output")
    
    submit_btn.click(
        fn=process_question,
        inputs=[file_input, user_prompt, system_prompt, cot_prompt, model_selector],
        outputs=[user_prompt_output, initial_response_output, thinking_output, reflection_output, final_output, system_prompt, cot_prompt]
    )

if __name__ == "__main__":
    iface.launch(share=False)
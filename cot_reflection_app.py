import gradio as gr
import os
import re
from cot_reflection import cot_reflection, cot_prompt

def process_question(user_prompt):
    system_prompt = "You are a legal assistant. Provide a detailed and accurate answer to the following question."
    try:
        result = cot_reflection(system_prompt=system_prompt, question=user_prompt, return_full_response=True, cot_prompt=cot_prompt)
        
        # Extract content for each section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', result, re.DOTALL)
        reflection_match = re.search(r'<reflection>(.*?)</reflection>', result, re.DOTALL)
        output_match = re.search(r'<output>(.*?)</output>', result, re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."
        reflection = reflection_match.group(1).strip() if reflection_match else "No reflection process provided."
        output = output_match.group(1).strip() if output_match else "No final output provided."
        
        # Assume initial_response is the same as output for this implementation
        initial_response = output

        return user_prompt, initial_response, thinking, reflection, output
    except Exception as e:
        return user_prompt, f"An error occurred: {str(e)}", "", "", ""

# Get the absolute path to the logo file
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "images", "Linklaters.svg.png")

# Gradio interface
with gr.Blocks() as iface:
    with gr.Column(scale=1):
        gr.Image(logo_path, show_label=False, height=70)
    
    # Add empty space
    gr.Markdown("<br><br>")
    
    gr.Markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>MVP for Chain of Thought Reflection Assistant</h1>
    <p style='text-align: center; font-style: italic; margin-top: 5px; font-size: 1.2em;'>powered by Linklaters GenAI Platform</p>
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Column(scale=1, min_width=600):
                user_prompt = gr.Textbox(
                    lines=2, 
                    label="",  # Set an empty label
                    placeholder="Ask a question and get a detailed answer using Chain of Thought reflection powered by Linklaters GenAI Platform."
                )
            submit_btn = gr.Button("Submit")
    
    with gr.Row():
        user_prompt_output = gr.Textbox(label="1. User Prompt")
        initial_response_output = gr.Textbox(label="2. Initial Response")
        thinking_output = gr.Textbox(label="3. Thinking")
        reflection_output = gr.Textbox(label="4. Reflection")
        final_output = gr.Textbox(label="5. Output")
    
    submit_btn.click(
        fn=process_question,
        inputs=user_prompt,
        outputs=[user_prompt_output, initial_response_output, thinking_output, reflection_output, final_output]
    )

if __name__ == "__main__":
    iface.launch()
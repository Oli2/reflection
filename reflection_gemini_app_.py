import gradio as gr
import os
from reflection_gemini import main as reflection_main, read_docx

# Default values
DEFAULT_PROJECT = "genai-sandbox-421407"
DEFAULT_LOCATION = "europe-west2"
DEFAULT_CREDENTIALS = "/Users/tomc/service_acccount_key.json"
DEFAULT_QUESTION = "Is this document about cooking?"
DEFAULT_MODEL = "gemini-1.5-pro"

def process_document(file, question, project, location, credentials):
    try:
        # Validate inputs
        if not file or not question:
            return "", "Please upload a document and provide a question.", "", "", ""
        
        # Set environment variables for GCP
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
        
        initial_answer, feedback, revised_answer = reflection_main(
            file.name, 
            question,
            DEFAULT_MODEL
        )
        
        if "Failed to generate" in initial_answer or "An error occurred" in initial_answer:
            return question, initial_answer, "", "", ""
        
        return question, initial_answer, feedback, revised_answer, read_docx(file.name)
    except Exception as e:
        return question, f"An error occurred: {str(e)}", "", "", ""

# Get the absolute path to the logo file
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "images", "Linklaters.svg.png")

# Verify logo file exists
if not os.path.exists(logo_path):
    print(f"Warning: Logo file not found at {logo_path}")

# Gradio interface
with gr.Blocks() as iface:
    # Use Gradio's Image component for the logo
    with gr.Row():
        with gr.Column():
            gr.Image(value=logo_path, 
                    show_label=False, 
                    container=False, 
                    height=70, 
                    show_download_button=False)
    
    gr.Markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0;'>Document Analysis with AI Chain-of-Thought</h1>
        <p style='text-align: center; font-style: italic; margin-top: 5px; font-size: 1.2em;'>
        Powered by Linklaters GenAI Platform
        </p>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            file_input = gr.File(label="Upload Document (DOCX)")
            question_input = gr.Textbox(
                lines=2, 
                label="Question",
                placeholder="Ask a question about the uploaded document.",
                value=DEFAULT_QUESTION
            )
            with gr.Accordion("GCP Settings", open=False):
                project_input = gr.Textbox(label="GCP Project ID", value=DEFAULT_PROJECT)
                location_input = gr.Textbox(label="GCP Location", value=DEFAULT_LOCATION)
                credentials_input = gr.Textbox(label="Path to GCP Credentials", value=DEFAULT_CREDENTIALS)
            submit_btn = gr.Button("Analyze Document")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            question_output = gr.Textbox(label="Question")
            initial_answer_output = gr.Textbox(label="Initial Answer")
            feedback_output = gr.Textbox(label="AI Reflection Feedback")
            revised_answer_output = gr.Textbox(label="Revised Answer")
            document_content_output = gr.Textbox(label="Document Content", visible=False)
    
    submit_btn.click(
        fn=process_document,
        inputs=[file_input, question_input, project_input, location_input, credentials_input],
        outputs=[question_output, initial_answer_output, feedback_output, revised_answer_output, document_content_output]
    )

if __name__ == "__main__":
    # Print the logo path for debugging
    print(f"Looking for logo at: {logo_path}")
    iface.launch(share=False)
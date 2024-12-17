import gradio as gr
import os
import re
import json
from datetime import datetime
from cot_reflection_file import (
    cot_reflection, 
    cot_prompt as default_cot_prompt, 
    system_prompt as default_system_prompt,
    get_model_response,
    AVAILABLE_MODELS
)
from document_utils import read_document
from db_utils import SnapshotDB

# Initialize database
db = SnapshotDB()

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

        return user_prompt, initial_response, actual_thinking, reflection, output, system_prompt, cot_prompt
    except Exception as e:
        print(f"Process error: {str(e)}")
        return user_prompt, f"An error occurred: {str(e)}", "", "", "", system_prompt, cot_prompt

def save_current_snapshot(snapshot_name, user_prompt, system_prompt, model_name, 
                         cot_prompt, initial_response, thinking, reflection, 
                         final_response, tags):
    try:
        if not snapshot_name:
            return "Error: Please provide a snapshot name!", None

        if not user_prompt:
            return "Error: No prompt to save!", None

        snapshot_data = {
            'snapshot_name': snapshot_name.strip(),
            'user_prompt': user_prompt,
            'system_prompt': system_prompt,
            'model_name': model_name,
            'cot_prompt': cot_prompt,
            'initial_response': initial_response,
            'thinking': thinking,
            'reflection': reflection,
            'final_response': final_response,
            'tags': tags.strip() if tags else ''
        }

        save_result = db.save_snapshot(snapshot_data)
        if "Error" in save_result or "Database error" in save_result:
            return save_result, None

        return "✓ Snapshot saved successfully!", update_snapshots_table()
    except Exception as e:
        print(f"Save error: {str(e)}")
        return f"Error saving snapshot: {str(e)}", None

def load_selected_snapshot(selected_data):
    try:
        if selected_data is None or len(selected_data) == 0:
            return [None] * 9 + ["Please select a snapshot to load"]
        
        snapshot_id = selected_data[0][0]  # Get ID from first selected row
        snapshot = db.get_snapshot_by_id(snapshot_id)
        
        if not snapshot:
            return [None] * 9 + ["Snapshot not found"]
        
        return [
            snapshot[1],  # snapshot_name
            snapshot[2],  # user_prompt
            snapshot[3],  # system_prompt
            snapshot[4],  # model_name
            snapshot[5],  # cot_prompt
            snapshot[6],  # initial_response
            snapshot[7],  # thinking
            snapshot[8],  # reflection
            snapshot[9],  # final_response
            "✓ Snapshot loaded successfully"
        ]
    except Exception as e:
        print(f"Load error: {str(e)}")
        return [None] * 9 + [f"Error loading snapshot: {str(e)}"]

def update_snapshots_table(search_term=None):
    try:
        snapshots = db.get_snapshots(search_term)
        if isinstance(snapshots, str) and ("Error" in snapshots or "Database error" in snapshots):
            return []
        return [[s[0], s[1], str(s[10]), s[4], s[2], s[11]] for s in snapshots]
    except Exception as e:
        print(f"Table update error: {str(e)}")
        return []

def delete_selected_snapshots(selected_data):
    try:
        if not selected_data or len(selected_data) == 0:
            return "Please select snapshots to delete", None
        
        snapshot_id = selected_data[0][0]
        result = db.delete_snapshot(snapshot_id)
        
        if "Error" in result or "Database error" in result:
            return result, None
            
        return "✓ Snapshot deleted successfully", update_snapshots_table()
    except Exception as e:
        print(f"Delete error: {str(e)}")
        return f"Error deleting snapshot: {str(e)}", None

def export_snapshots():
    try:
        result = db.export_snapshots()
        if "Error" in result or "Database error" in result:
            return result
        return "✓ Snapshots exported successfully"
    except Exception as e:
        print(f"Export error: {str(e)}")
        return f"Error exporting snapshots: {str(e)}"

# Get the absolute path to the logo file
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "images", "Linklaters.svg.png")

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    with gr.Column(scale=1):
        gr.Image(logo_path, show_label=False, height=70)
    
    gr.Markdown("<br><br>")
    
    gr.Markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>Document Analysis with Chain of Thought Reflection</h1>
    <p style='text-align: center; font-style: italic; margin-top: 5px; font-size: 1.2em;'>powered by Linklaters GenAI Platform</p>
    """)
    
    with gr.Tabs():
        with gr.TabItem("Analysis"):
            with gr.Row():
                with gr.Column():
                    with gr.Column(scale=1, min_width=600):
                        model_selector = gr.Dropdown(
                            choices=list(AVAILABLE_MODELS.keys()),
                            value=list(AVAILABLE_MODELS.keys())[0],
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
                        with gr.Accordion("Chain-of-Thought Prompt", open=False):
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
                    submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Row():
                user_prompt_output = gr.Textbox(label="1. User Prompt", interactive=False)
                initial_response_output = gr.Textbox(label="2. Initial Response", interactive=False)
                thinking_output = gr.Textbox(label="3. Thinking", interactive=False)
                reflection_output = gr.Textbox(label="4. Reflection", interactive=False)
                final_output = gr.Textbox(label="5. Final Output", interactive=False)
            
            with gr.Row():
                snapshot_name = gr.Textbox(
                    label="Snapshot Name",
                    placeholder="Enter a name for this snapshot"
                )
                tags_input = gr.Textbox(
                    label="Tags (comma-separated)",
                    placeholder="tag1, tag2, tag3"
                )
                save_btn = gr.Button("💾 Save Snapshot", variant="secondary")
            
            with gr.Row():
                snapshot_status = gr.Textbox(
                    label="Status",
                    interactive=False
                )

        with gr.TabItem("Saved Snapshots"):
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search Snapshots",
                    placeholder="Search by name, prompt, or tags..."
                )
            
            snapshots_table = gr.Dataframe(
                headers=["ID", "Name", "Created At", "Model", "Prompt", "Tags"],
                interactive=True,
                label="Saved Snapshots",
                value=update_snapshots_table(),
                type="array"
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                load_btn = gr.Button("📂 Load Selected", variant="primary")
                delete_btn = gr.Button("🗑️ Delete Selected", variant="secondary")
                export_btn = gr.Button("📤 Export", variant="secondary")
            
            with gr.Row():
                operation_status = gr.Textbox(
                    label="Operation Status",
                    interactive=False
                )
    
    # Connect components
    submit_btn.click(
        fn=process_question,
        inputs=[file_input, user_prompt, system_prompt, cot_prompt, model_selector],
        outputs=[user_prompt_output, initial_response_output, thinking_output, 
                reflection_output, final_output, system_prompt, cot_prompt]
    )
    
    save_btn.click(
        fn=save_current_snapshot,
        inputs=[snapshot_name, user_prompt_output, system_prompt, 
                model_selector, cot_prompt, initial_response_output,
                thinking_output, reflection_output, final_output, tags_input],
        outputs=[snapshot_status, snapshots_table]
    )
    
    search_box.change(
        fn=update_snapshots_table,
        inputs=[search_box],
        outputs=snapshots_table
    )
    
    refresh_btn.click(
        fn=update_snapshots_table,
        inputs=[search_box],
        outputs=snapshots_table
    )
    
    load_btn.click(
        fn=load_selected_snapshot,
        inputs=[snapshots_table],
        outputs=[
            snapshot_name, user_prompt, system_prompt, model_selector,
            cot_prompt, initial_response_output, thinking_output,
            reflection_output, final_output, operation_status
        ]
    )
    
    delete_btn.click(
        fn=delete_selected_snapshots,
        inputs=[snapshots_table],
        outputs=[operation_status, snapshots_table]
    )
    
    export_btn.click(
        fn=export_snapshots,
        inputs=[],
        outputs=operation_status
    )

if __name__ == "__main__":
    iface.launch(share=False)
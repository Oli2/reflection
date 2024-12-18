import gradio as gr
import os
import re
import json
from datetime import datetime
from typing import Optional, List, Any, Dict, Tuple
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
        initial_response_prompt = (f"{system_prompt}\n\n{doc_content}"
                                   f"Question: {user_prompt}\n\n"
                                   "Provide a concise answer to this question without any explanation or reasoning.")
        initial_response = get_model_response(selected_model, initial_response_prompt)

        return user_prompt, initial_response, actual_thinking, reflection, output, system_prompt, cot_prompt
    except Exception as e:
        print(f"Process error: {str(e)}")
        return user_prompt, f"An error occurred: {str(e)}", "", "", "", system_prompt, cot_prompt

def load_snapshot_by_id(snapshot_id: str) -> List[Optional[Any]]:
    """
    Load a snapshot by ID and update UI components.
    
    Args:
        snapshot_id: ID of the snapshot to load
        
    Returns:
        List of values for Gradio components in correct order
    """
    try:
        if not snapshot_id:
            return [None] * 9 + ["Please enter a snapshot ID to load"]
        
        try:
            snapshot_id_int = int(snapshot_id)
        except ValueError:
            return [None] * 9 + ["Invalid Snapshot ID. Please enter a numeric ID."]
        
        # Get snapshot data from database
        snapshot_data = db.get_snapshot_by_id(snapshot_id_int)
        
        if not snapshot_data:
            return [None] * 9 + ["Snapshot not found"]
            
        # Extract values from snapshot data
        return [
            snapshot_data.get("snapshot_name", ""),          # Snapshot name
            snapshot_data.get("user_prompt", ""),            # User prompt
            snapshot_data.get("system_prompt", ""),          # System prompt
            snapshot_data.get("model_name", ""),             # Model name
            snapshot_data.get("cot_prompt", ""),             # Chain of thought prompt
            snapshot_data.get("initial_response", ""),       # Initial response
            snapshot_data.get("thinking", ""),               # Thinking process
            snapshot_data.get("reflection", ""),             # Reflection
            snapshot_data.get("final_response", ""),         # Final response
            "✓ Snapshot loaded successfully"                 # Status message
        ]
    except Exception as e:
        print(f"Load error: {str(e)}")
        return [None] * 9 + [f"Error loading snapshot: {str(e)}"]

def update_snapshots_table(search_term: str = "") -> List[List]:
    """
    Update the snapshots table with filtered results.
    
    Args:
        search_term: Optional search term to filter snapshots
        
    Returns:
        List of snapshot data for table display
    """
    snapshots = db.get_snapshots(search_term)
    return [[s[0], s[1], s[10], s[4], s[2], s[11]] for s in snapshots]

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    with gr.Tabs():
        # Analysis Tab
        with gr.TabItem("Analysis"):
            with gr.Row():
                with gr.Column():
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
                        label="User Prompt",
                        placeholder="Ask a question about the uploaded document..."
                    )
                    with gr.Accordion("System and Chain-of-Thought Prompts", open=False):
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

        # Saved Snapshots Tab
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
                type="array",
                datatype=["number", "str", "str", "str", "str", "str"]
            )
            
            with gr.Row():
                snapshot_id_input = gr.Textbox(
                    label="Enter Snapshot ID to Load",
                    placeholder="Enter the ID of the snapshot you want to load"
                )
                load_btn = gr.Button("📂 Load Snapshot by ID", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
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
        fn=lambda *args: (
            db.save_snapshot({
                'snapshot_name': args[0],
                'user_prompt': args[1],
                'system_prompt': args[2],
                'model_name': args[3],
                'cot_prompt': args[4],
                'initial_response': args[5],
                'thinking': args[6],
                'reflection': args[7],
                'final_response': args[8],
                'tags': args[9]
            }),
            update_snapshots_table()
        ),
        inputs=[snapshot_name, user_prompt_output, system_prompt, 
                model_selector, cot_prompt, initial_response_output,
                thinking_output, reflection_output, final_output, tags_input],
        outputs=[snapshot_status, snapshots_table]
    )
    
    load_btn.click(
        fn=load_snapshot_by_id,
        inputs=[snapshot_id_input],
        outputs=[
            snapshot_name,           # Snapshot name
            user_prompt,             # User prompt
            system_prompt,           # System prompt
            model_selector,          # Model name
            cot_prompt,              # Chain of thought prompt
            initial_response_output, # Initial response
            thinking_output,         # Thinking process
            reflection_output,       # Reflection
            final_output,           # Final response
            operation_status        # Status message
        ]
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
    
    delete_btn.click(
        fn=lambda *args: db.delete_selected_snapshots(*args),
        inputs=[snapshots_table],
        outputs=[operation_status, snapshots_table]
    )
    
    export_btn.click(
        fn=lambda: db.export_snapshots(),
        inputs=[],
        outputs=operation_status
    )

if __name__ == "__main__":
    iface.launch(share=False)
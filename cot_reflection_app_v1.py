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

def get_available_models() -> List[str]:
    """
    Get list of available models.
    
    Returns:
        List of model names
    """
    return list(AVAILABLE_MODELS.keys())

def process_question(file, user_prompt, system_prompt, cot_prompt, selected_model, use_default_cot):
    """
    Process user question using selected model and prompts.
    
    Args:
        file: Optional document file
        user_prompt: User's question
        system_prompt: System context
        cot_prompt: Chain of thought prompt
        selected_model: Name of selected model
        use_default_cot: Boolean indicating if default CoT prompt should be used
        
    Returns:
        Tuple of processed outputs
    """
    try:
        # Validate model selection
        if selected_model not in AVAILABLE_MODELS:
            raise ValueError(f"Invalid model selected: {selected_model}")
            
        # Read document content if file is provided
        document_content = None
        if file is not None:
            document_content = read_document(file.name)

        # If the checkbox is checked, use CoT logic
        if use_default_cot:
            # If the checkbox is checked, generate an initial response without CoT
            doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
            initial_response_prompt = (f"{system_prompt}\n\n{doc_content}"
                                       f"Question: {user_prompt}\n\n"
                                       "Provide a concise answer to this question without any explanation or reasoning.")
            initial_response = get_model_response(selected_model, initial_response_prompt)            
            # Get thinking, reflection, and output from cot_reflection
            thinking, reflection, output = cot_reflection(
                system_prompt=system_prompt,
                cot_prompt=default_cot_prompt,  # Use default CoT prompt
                question=user_prompt,
                document_content=document_content,
                model_name=selected_model
            )

            # Extract the actual thinking content
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking, re.DOTALL)
            actual_thinking = thinking_match.group(1).strip() if thinking_match else thinking

            # Return all outputs related to CoT
            return user_prompt, initial_response, actual_thinking, reflection, output, system_prompt, default_cot_prompt
        else:
            # If the checkbox is not checked, generate a response without CoT
            doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
            initial_response_prompt = (f"{system_prompt}\n\n{doc_content}"
                                       f"Question: {user_prompt}\n\n"
                                       "Provide a concise answer to this question without any explanation or reasoning.")
            initial_response = get_model_response(selected_model, initial_response_prompt)

            # Return only the user prompt and initial response, with empty strings for CoT outputs
            return user_prompt, initial_response, "", "", "", system_prompt, None  # No CoT prompt used, Final Output as empty string

    except Exception as e:
        print(f"Process error: {str(e)}")
        return user_prompt, f"An error occurred: {str(e)}", "", "", "", None, None  # No CoT prompt used, Final Output as empty string

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
    Returns data in the format: [ID, Name, Created At, Model, Prompt, Tags]
    """
    snapshots = db.get_snapshots(search_term)
    return [[s[0], s[1], s[2], s[3], s[4], s[5]] for s in snapshots]

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    with gr.Tabs():
        # Analysis Tab
        with gr.TabItem("Analysis"):
            with gr.Row():
                with gr.Column():
                    model_selector = gr.Dropdown(
                        choices=get_available_models(),
                        value="Gemini 2.0 Flash",
                        label="Select Model",
                        interactive=True,
                        info="Choose from the dropdown menu of the available LLMs"
                    )
                    
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=["pdf", "docx"]
                    )
                    
                    user_prompt = gr.Textbox(
                        lines=2,
                        label="User Prompt",
                        placeholder="Ask a question about the uploaded document..."
                    )
                    
                    use_default_cot = gr.Checkbox(
                        label="Use Default Chain of Thought Prompt",
                        value=False
                    )
                    
                    submit_btn = gr.Button("Submit", variant="primary")
                    
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

            with gr.Row():
                user_prompt_output = gr.Textbox(label="1. User Prompt")
                initial_response_output = gr.Textbox(label="2. Initial Response")
                thinking_output = gr.Textbox(label="3. Thinking")
                reflection_output = gr.Textbox(label="4. Reflection")
                final_output = gr.Textbox(label="5. Final Output")

            with gr.Row():
                snapshot_name = gr.Textbox(
                    label="Snapshot Name",
                    placeholder="Enter a name for this snapshot"
                )
                tags_input = gr.Textbox(
                    label="Tags",
                    placeholder="tag1, tag2, tag3"
                )
                save_btn = gr.Button("💾 Save", variant="secondary")

            with gr.Row():
                snapshot_status = gr.Textbox(label="Status")

        # Saved Snapshots Tab
        with gr.TabItem("Saved Snapshots"):
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Search snapshots..."
                )
            
            snapshots_table = gr.Dataframe(
                headers=["ID", "Name", "Created At", "Model", "Prompt", "Tags"],
                label="Saved Snapshots",
                value=update_snapshots_table(),
                wrap=True,
                row_count=5
            )
            
            with gr.Row():
                # Using gr.Number() for integer input
                snapshot_id_input = gr.Number(
                    label="Snapshot ID",
                    precision=0,
                    minimum=1,
                    step=1
                )
                
                with gr.Row():
                    load_btn = gr.Button("📂 Load", variant="primary")
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    delete_btn = gr.Button("🗑️ Delete", variant="secondary")
                    export_btn = gr.Button("📤 Export", variant="secondary")
            
            operation_status = gr.Textbox(label="Status")

    # Connect components
    submit_btn.click(
        fn=process_question,
        inputs=[
            file_input, user_prompt, system_prompt, 
            cot_prompt, model_selector, use_default_cot
        ],
        outputs=[
            user_prompt_output, initial_response_output, 
            thinking_output, reflection_output, final_output, 
            system_prompt, cot_prompt
        ]
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
        inputs=[
            snapshot_name, user_prompt_output, system_prompt, 
            model_selector, cot_prompt, initial_response_output,
            thinking_output, reflection_output, final_output, 
            tags_input
        ],
        outputs=[snapshot_status, snapshots_table]
    )
    
    delete_btn.click(
        fn=lambda snapshot_id: db.delete_snapshot(int(snapshot_id)) if snapshot_id is not None else ("Please enter a snapshot ID", update_snapshots_table()),
        inputs=[snapshot_id_input],
        outputs=[operation_status, snapshots_table]
    )
    
    refresh_btn.click(
        fn=update_snapshots_table,
        inputs=[search_box],
        outputs=snapshots_table
    )
    
    search_box.change(
        fn=update_snapshots_table,
        inputs=[search_box],
        outputs=snapshots_table
    )
    
    export_btn.click(
        fn=lambda: (db.export_snapshots(), "Export completed successfully")[1],
        outputs=operation_status
    )

if __name__ == "__main__":
    iface.launch(share=True)
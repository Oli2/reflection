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
    AVAILABLE_MODELS,
    get_model_params
)
from db_utils import SnapshotDB
import PyPDF2
from docx import Document

# Initialize database
db = SnapshotDB()

def get_available_models() -> List[str]:
    """
    Get list of available models.
    
    Returns:
        List of model names
    """
    return list(AVAILABLE_MODELS.keys())

def process_question(file, user_prompt, system_prompt, cot_prompt, selected_model, use_default_cot, temperature, top_p):
    """
    Process user question using selected model and prompts.
    
    Args:
        file: Optional document file
        user_prompt: User's question
        system_prompt: System context (can be default or customized)
        cot_prompt: Chain of thought prompt (can be default or customized)
        selected_model: Name of selected model
        use_default_cot: Boolean indicating if CoT processing should be used
        temperature: Temperature parameter for response generation
        top_p: Top-p parameter for response generation
        
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
            try:
                import io
                file_obj = io.BytesIO(file)
                
                try:
                    pdf_reader = PyPDF2.PdfReader(file_obj)
                    document_content = '\n'.join(page.extract_text() for page in pdf_reader.pages)
                except:
                    file_obj.seek(0)
                    doc = Document(file_obj)
                    document_content = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
            except Exception as e:
                raise ValueError("Error reading document. Please ensure it's a valid PDF or DOCX file.")

        # Prepare document content string if document was provided
        doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
        
        if use_default_cot:
            # Generate initial response without reasoning
            initial_response_prompt = (f"{system_prompt}\n\n{doc_content}"
                                     f"Question: {user_prompt}\n\n"
                                     "Provide a concise answer to this question without any explanation or reasoning.")
            initial_response = get_model_response(selected_model, initial_response_prompt, temperature, top_p)
            
            # Use CoT processing with the current cot_prompt (either default or customized)
            thinking, reflection, output = cot_reflection(
                system_prompt=system_prompt,
                cot_prompt=cot_prompt,  # This will be either default or customized version
                question=user_prompt,
                document_content=document_content,
                model_name=selected_model,
                temperature=temperature,
                top_p=top_p
            )

            # Extract the actual thinking content
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking, re.DOTALL)
            actual_thinking = thinking_match.group(1).strip() if thinking_match else thinking

            # Return full CoT processing results
            return user_prompt, initial_response, actual_thinking, reflection, output, system_prompt, cot_prompt
            
        else:
            # When use_default_cot is False, only use system prompt without CoT
            direct_response_prompt = (f"{system_prompt}\n\n{doc_content}"
                                    f"Question: {user_prompt}\n\n"
                                    "Analyze the question and provide a comprehensive answer.")
            
            direct_response = get_model_response(selected_model, direct_response_prompt, temperature, top_p)
            
            # Return response without CoT components
            return user_prompt, direct_response, "", "", "", system_prompt, None

    except Exception as e:
        return user_prompt, f"An error occurred: {str(e)}", "", "", "", None, None

def load_snapshot_by_id(snapshot_id: str) -> List[Optional[Any]]:
    """
    Load a snapshot by ID and update UI components.
    
    Args:
        snapshot_id: ID of the snapshot to load
        
    Returns:
        List of values for Gradio components in correct order:
        [snapshot_name, user_prompt, system_prompt, model_name, cot_prompt,
         initial_response, thinking, reflection, final_response, status_message]
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
        return [None] * 9 + [f"Error loading snapshot: {str(e)}"]

def update_snapshots_table(search_term: str = "") -> List[List]:
    """
    Update the snapshots table with filtered results.
    Returns data in the format: [ID, Name, Created At, Model, Prompt, Tags]
    """
    snapshots = db.get_snapshots(search_term)
    return [[s[0], s[1], s[2], s[3], s[4], s[5]] for s in snapshots]

def export_snapshot(snapshot_id: int) -> str:
    """
    Export a single snapshot as JSON and return its content.
    
    Args:
        snapshot_id: ID of the snapshot to export
        
    Returns:
        JSON string of the snapshot content
    """
    try:
        if not snapshot_id:
            return "Please enter a snapshot ID to export"
            
        # Get snapshot data from database
        snapshot_data = db.get_snapshot_by_id(int(snapshot_id))
        
        if not snapshot_data:
            return "Snapshot not found"
            
        # Convert snapshot to formatted JSON string
        json_content = json.dumps(snapshot_data, indent=2, ensure_ascii=False)
        
        # Return JSON content to be displayed in popup
        return json_content
        
    except Exception as e:
        return f"Error exporting snapshot: {str(e)}"

def default_evaluation_prompt():
    return """Please evaluate the following two responses based on the specified metrics.
    For each metric, provide a score from 1-10 and detailed justification.
    
    Evaluation Aspects:
    1. Clarity: How clear and understandable is the response?
    2. Completeness: How thoroughly does it address the topic?
    3. Accuracy: How accurate and reliable is the information?
    4. Reasoning Quality: How well-structured and logical is the reasoning?
    5. Practical Applicability: How useful is this response in practice?
    
    Provide:
    1. Numerical scores for each metric
    2. Detailed qualitative analysis
    3. Strengths and weaknesses comparison
    4. Overall recommendation
    """

def create_evaluation_prompt(content1: str, content2: str, metrics: List[str], custom_criteria: str, model1_name: str, model2_name: str) -> str:
    """Create the evaluation prompt for the judge model with model names"""
    return f"""
    {custom_criteria}

    === Response by {model1_name} ===
    {content1}

    === Response by {model2_name} ===
    {content2}

    Please evaluate these responses focusing on these specific metrics: {', '.join(metrics)}
    When referring to the responses in your analysis, use "Response by {model1_name}" and "Response by {model2_name}" respectively.
    
    System / Instruction to Judge LLM:
    You are an impartial expert evaluator. Your task is to assess and compare these responses on several metrics.
    Follow these guidelines:
    1. Always refer to responses as "Response by {model1_name}" and "Response by {model2_name}"
    2. Provide scores and detailed justifications for each metric
    3. Compare strengths and weaknesses
    4. Give an overall recommendation
    
    Required Output Format:
    1. Summaries
       - Response by {model1_name}: [Summary]
       - Response by {model2_name}: [Summary]
    
    2. Scores and Justifications
       For each metric:
       - Response by {model1_name}: [Score]/10 ([Justification])
       - Response by {model2_name}: [Score]/10 ([Justification])
    
    3. Strengths and Weaknesses Comparison
       - Response by {model1_name}: [Strengths] and [Areas for Improvement]
       - Response by {model2_name}: [Strengths] and [Areas for Improvement]
    
    4. Overall Recommendation
       [Clear statement of which response is stronger, with brief rationale]
    """

def load_snapshot_previews(snapshot1_id: int, snapshot2_id: int, aspects: List[str]) -> Tuple[str, str, str, str]:
    """
    Load and format selected aspects of two snapshots for preview.
    Returns tuple of (content1, content2, model1_name, model2_name)
    """
    try:
        if not snapshot1_id or not snapshot2_id:
            return "", "", "", ""
            
        # Get snapshots from database
        snap1 = db.get_snapshot_by_id(int(snapshot1_id))
        snap2 = db.get_snapshot_by_id(int(snapshot2_id))
        
        if not snap1 or not snap2:
            return "Snapshot not found", "Snapshot not found", "", ""
        
        # Get model names from snapshots
        model1_name = snap1.get('model_name', 'Unknown Model')
        model2_name = snap2.get('model_name', 'Unknown Model')
        
        # Format previews
        def format_snapshot(snap: Dict) -> str:
            preview = ""
            for aspect in aspects:
                if aspect == "Thinking":
                    preview += f"=== Thinking ===\n{snap.get('thinking', '')}\n\n"
                elif aspect == "Reflection":
                    preview += f"=== Reflection ===\n{snap.get('reflection', '')}\n\n"
                elif aspect == "Final Output":
                    preview += f"=== Final Output ===\n{snap.get('final_response', '')}\n\n"
            return preview.strip()
        
        return format_snapshot(snap1), format_snapshot(snap2), model1_name, model2_name
        
    except Exception as e:
        return f"Error loading preview: {str(e)}", f"Error loading preview: {str(e)}", "", ""

def update_evaluation(
    snapshot1_id: int,
    snapshot2_id: int,
    aspects: List[str],
    judge_model: str,
    predefined_metrics: List[str],
    custom_criteria: str,
    judge_temperature: float,
    judge_top_p: float,
    progress=gr.Progress()
) -> str:  # Return type is now just str since we have single output
    """
    Performs the evaluation with progress updates.
    Returns formatted evaluation summary.
    """
    try:
        # Step 1: Load snapshots (20%)
        progress(0.2, desc="Loading snapshots...")
        content1, content2, model1_name, model2_name = load_snapshot_previews(
            snapshot1_id, snapshot2_id, aspects
        )
        if not content1 or not content2:
            return "Error: Failed to load snapshots"
        
        # Step 2: Create evaluation prompt (40%)
        progress(0.4, desc="Creating evaluation prompt...")
        eval_prompt = create_evaluation_prompt(
            content1, content2, predefined_metrics, 
            custom_criteria, model1_name, model2_name
        )
        
        # Step 3: Get model response (70%)
        progress(0.7, desc="Getting model evaluation...")
        evaluation = get_model_response(
            judge_model,
            eval_prompt,
            judge_temperature,
            judge_top_p
        )
        
        # Step 4: Format response (90%)
        progress(0.9, desc="Formatting evaluation...")
        formatted_eval = f"""
=================================================================
                         EVALUATION SUMMARY                         
=================================================================

MODELS COMPARED
-----------------------------------------------------------------
• {model1_name} (Snapshot {snapshot1_id})
• {model2_name} (Snapshot {snapshot2_id})

EVALUATION SETTINGS
-----------------------------------------------------------------
• Judge Model: {judge_model}
• Temperature: {judge_temperature}
• Top-p: {judge_top_p}

EVALUATION SCOPE
-----------------------------------------------------------------
• Metrics Evaluated: {', '.join(predefined_metrics)}
• Aspects Compared: {', '.join(aspects)}

ANALYSIS
-----------------------------------------------------------------
{evaluation}

=================================================================
"""
        
        # Step 5: Complete (100%)
        progress(1.0, desc="Evaluation complete!")
        return formatted_eval
        
    except Exception as e:
        error_message = f"""
=================================================================
                         EVALUATION ERROR                          
=================================================================

An error occurred during evaluation: {str(e)}

=================================================================
"""
        return error_message

# Add this function definition before the Gradio interface
def update_param_ranges(model_name):
    """Update parameter ranges based on selected model"""
    params = get_model_params(model_name)
    return [
        gr.update(minimum=params["temp_range"][0], maximum=params["temp_range"][1]),
        gr.update(minimum=params["top_p_range"][0], maximum=params["top_p_range"][1])
    ]

# Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".small-font-table { font-size: 0.9em !important; }"
) as iface:
    # Add logo and title at the top
    with gr.Row():
        gr.Image(
            value="./images/Linklaters.svg.png",
            height=30,
            show_label=False,
            container=False
        )
    
    gr.Markdown(
        "<div style='text-align: center; color: magenta; margin-bottom: 20px;'>"
        "Contract Analysis Workbench Prototype"
        "</div>"
    )

    with gr.Tabs():
        with gr.TabItem("Prompt Wizard"):
            # First row for model selection and parameters side by side
            with gr.Row():
                # Model selector column
                with gr.Column(scale=1):
                    with gr.Group():  # Changed from gr.Box() to gr.Group()
                        model_selector = gr.Dropdown(
                            choices=get_available_models(),
                            value="Gemini 2.0 Flash",
                            label="Select Model",
                            interactive=True,
                            info="Choose from the dropdown menu of the available LLMs",
                            container=True
                        )
                # Model parameters column
                with gr.Column(scale=1):
                    with gr.Group():  # Changed from gr.Box() to gr.Group()
                        with gr.Row():
                            temperature = gr.Number(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.2,
                                step=0.1,
                                label="Temperature",
                                info="Controls randomness (0.0 = deterministic, 1.0 = creative)"
                            )
                            top_p = gr.Number(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                label="Top-p",
                                info="Controls diversity (lower = more focused)"
                            )

            # Second main column for other inputs
            with gr.Column(scale=1):
                with gr.Accordion("Upload Document (Optional)", open=False):
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=["pdf", "docx"],
                        type="binary"
                    )
                
                system_prompt = gr.Textbox(
                    lines=2,
                    label="System Prompt",
                    value=default_system_prompt
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
                
                # 2. Rename accordion and remove system prompt from it
                with gr.Accordion("Chain-of-Thought Prompt", open=False):
                    cot_prompt = gr.Textbox(
                        lines=4,
                        label="Chain of Thought Prompt",
                        value=default_cot_prompt
                    )

            with gr.Row():
                user_prompt_output = gr.Textbox(label="1. User Prompt")
                initial_response_output = gr.Textbox(label="2. Initial Response")
                thinking_output = gr.Textbox(label="3. Thinking", show_copy_button=True)
                reflection_output = gr.Textbox(label="4. Reflection", show_copy_button=True)
                final_output = gr.Textbox(label="5. Final Output", show_copy_button=True)

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
                    export_btn = gr.Button("📤 Share", variant="secondary")
            
            # JSON output
            json_output = gr.JSON(
                label="Snapshot Content",
                visible=False  # Initially hidden
            )
            
            operation_status = gr.Textbox(label="Status")

        # Snapshot Evaluator Tab
        with gr.TabItem("Snapshot Evaluator"):
            with gr.Row():
                search_box_eval = gr.Textbox(
                    label="Search Snapshots",
                    placeholder="Search snapshots..."
                )
            
            snapshots_table_eval = gr.Dataframe(
                headers=["ID", "Name", "Created At", "Model", "Prompt", "Tags"],
                label="Available Snapshots",
                value=update_snapshots_table(),
                wrap=True,
                row_count=5,
                elem_classes="small-font-table"
            )

            with gr.Row():
                snapshot1_id = gr.Number(
                    label="First Snapshot ID",
                    precision=0,
                    value=None
                )
                snapshot2_id = gr.Number(
                    label="Second Snapshot ID",
                    precision=0,
                    value=None
                )
            
            # Add comparison aspects selection with all options selected by default
            comparison_aspects = gr.CheckboxGroup(
                choices=["Thinking", "Reflection", "Final Output"],
                label="Select Aspects to Compare",
                value=["Thinking", "Reflection", "Final Output"]
            )

            # Model and parameters selection - directly below aspects
            with gr.Row():
                with gr.Column(scale=1):
                    judge_model = gr.Dropdown(
                        choices=get_available_models(),
                        label="Judge LLM",
                        value="Gemini 2.0 Flash"
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        judge_temperature = gr.Number(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            label="Temperature"
                        )
                        judge_top_p = gr.Number(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.95,
                            label="Top-p"
                        )

            # Preview section with model names
            with gr.Row():
                with gr.Column():
                    model1_name = gr.Textbox(
                        label="Model 1",
                        interactive=False
                    )
                    preview1 = gr.TextArea(
                        label="First Snapshot Preview",
                        interactive=False
                    )
                with gr.Column():
                    model2_name = gr.Textbox(
                        label="Model 2",
                        interactive=False
                    )
                    preview2 = gr.TextArea(
                        label="Second Snapshot Preview",
                        interactive=False
                    )

            with gr.Accordion("Evaluation Criteria", open=False):
                predefined_metrics = gr.CheckboxGroup(
                    choices=[
                        "Clarity (1-10)",
                        "Completeness (1-10)",
                        "Accuracy (1-10)",
                        "Reasoning Quality (1-10)",
                        "Practical Applicability (1-10)"
                    ],
                    label="Predefined Metrics",
                    value=[
                        "Clarity (1-10)",
                        "Completeness (1-10)",
                        "Accuracy (1-10)",
                        "Reasoning Quality (1-10)",
                        "Practical Applicability (1-10)"
                    ]
                )
                
                custom_criteria = gr.Textbox(
                    label="Custom Evaluation Instructions",
                    value=default_evaluation_prompt()
                )

            evaluate_btn = gr.Button("Evaluate", variant="primary")

            # Evaluation results - single textbox for evaluation summary
            evaluation_summary = gr.Textbox(
                label="Evaluation Summary",
                value="",
                show_copy_button=True,
                lines=20,
                interactive=False
            )

            # Connect row selection from table to snapshot IDs
            def select_snapshot(evt: gr.SelectData):
                selected_row = evt.index[0]
                return selected_row + 1  # Assuming IDs start from 1

            snapshots_table_eval.select(
                fn=select_snapshot,
                inputs=[],
                outputs=[snapshot1_id]
            )

            # Connect preview updates
            def update_previews(id1, id2, aspects):
                if not id1 or not id2 or not aspects:
                    return "", "", "", ""
                return load_snapshot_previews(id1, id2, aspects)

            # Connect the preview updates to relevant input changes
            for component in [snapshot1_id, snapshot2_id, comparison_aspects]:
                component.change(
                    fn=update_previews,
                    inputs=[snapshot1_id, snapshot2_id, comparison_aspects],
                    outputs=[preview1, preview2, model1_name, model2_name]
                )

            # Connect the evaluate button
            evaluate_btn.click(
                fn=update_evaluation,
                inputs=[
                    snapshot1_id,
                    snapshot2_id,
                    comparison_aspects,
                    judge_model,
                    predefined_metrics,
                    custom_criteria,
                    judge_temperature,
                    judge_top_p
                ],
                outputs=[evaluation_summary]  # Single output for evaluation summary
            )

    # Connect components
    submit_btn.click(
        fn=process_question,
        inputs=[
            file_input, user_prompt, system_prompt, 
            cot_prompt, model_selector, use_default_cot,
            temperature, top_p
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
    
    # Update the export button click handler
    def handle_export(snapshot_id):
        """
        Handle the export button click.
        
        Args:
            snapshot_id: ID of the snapshot to export
            
        Returns:
            Tuple of (json_content, status_message)
        """
        if not snapshot_id:
            return gr.update(visible=False, value=None), "Please enter a snapshot ID to export"
        try:
            json_content = export_snapshot(snapshot_id)
            # Try to parse the JSON string to ensure it's valid
            parsed_json = json.loads(json_content)
            return gr.update(visible=True, value=parsed_json), "Export successful"
        except Exception as e:
            return gr.update(visible=False, value=None), f"Export failed: {str(e)}"

    export_btn.click(
        fn=handle_export,
        inputs=[snapshot_id_input],
        outputs=[
            json_output,
            operation_status
        ]
    )

    # Add the load button click event handler
    load_btn.click(
        fn=load_snapshot_by_id,
        inputs=[snapshot_id_input],
        outputs=[
            snapshot_name,
            user_prompt,
            system_prompt,
            model_selector,
            cot_prompt,
            initial_response_output,
            thinking_output,
            reflection_output,
            final_output,
            operation_status
        ]
    )

    # Update parameter ranges when judge model changes
    judge_model.change(
        fn=update_param_ranges,
        inputs=[judge_model],
        outputs=[judge_temperature, judge_top_p]
    )

    # Connect search box to table updates
    search_box_eval.change(
        fn=update_snapshots_table,
        inputs=[search_box_eval],
        outputs=[snapshots_table_eval]
    )

    # Add custom CSS for clearer markdown rendering and button styling.
    gr.Markdown("""
        <style>
        .evaluation-results {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            padding: 25px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 20px 0;
            width: 100%;
            box-sizing: border-box;
        }
        .status-message {
            color: #2196F3;
            font-weight: bold;
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
        }
        </style>
    """)

if __name__ == "__main__":
    iface.launch(share=False)
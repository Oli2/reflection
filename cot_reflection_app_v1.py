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

def process_question(file, user_prompt, system_prompt, cot_prompt, selected_model, use_default_cot):
    """
    Process user question using selected model and prompts.
    
    Args:
        file: Optional document file
        user_prompt: User's question
        system_prompt: System context (can be default or customized)
        cot_prompt: Chain of thought prompt (can be default or customized)
        selected_model: Name of selected model
        use_default_cot: Boolean indicating if CoT processing should be used
        
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
            initial_response = get_model_response(selected_model, initial_response_prompt)
            
            # Use CoT processing with the current cot_prompt (either default or customized)
            thinking, reflection, output = cot_reflection(
                system_prompt=system_prompt,
                cot_prompt=cot_prompt,  # This will be either default or customized version
                question=user_prompt,
                document_content=document_content,
                model_name=selected_model
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
            
            direct_response = get_model_response(selected_model, direct_response_prompt)
            
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

# def create_evaluation_prompt(content1: str, content2: str, metrics: List[str], custom_criteria: str) -> str:
#     """Create the evaluation prompt for the judge model"""
#     return f"""
#     {custom_criteria}

#     === FIRST RESPONSE ===
#     {content1}

#     === SECOND RESPONSE ===
#     {content2}

#     Please evaluate these responses focusing on these specific metrics: {', '.join(metrics)}
    
#     For each metric, provide:
#     1. A numerical score (1-10)
#     2. Detailed justification with specific examples
#     3. Suggestions for improvement

#     Format your response as follows:
    
#     NUMERICAL SCORES:
#     - Metric1: X/10
#     - Metric2: Y/10
#     ...

#     QUALITATIVE ANALYSIS:
#     [Your detailed analysis here]

#     COMPARISON SUMMARY:
#     - Strengths of First Response:
#     - Strengths of Second Response:
#     - Areas for Improvement (First):
#     - Areas for Improvement (Second):

#     OVERALL RECOMMENDATION:
#     [Your final recommendation]
#     """
def create_evaluation_prompt(content1: str, content2: str, metrics: List[str], custom_criteria: str) -> str:
    """Create the evaluation prompt for the judge model"""
    return f"""
    {custom_criteria}

    === RESPONSE A ===
    {content1}

    === RESPONSE B===
    {content2}

    Please evaluate these responses focusing on these specific metrics: {', '.join(metrics)}
    

System / Instruction to Judge LLM:
You are an impartial expert evaluator. You will be given two responses (Response A and Response B) to a prompt. Your task is to assess and compare these responses on several metrics. Follow the instructions below carefully, using a detailed internal chain-of-thought to arrive at your conclusions. However, only provide your final, summarized justifications in your written output—do not reveal the entire chain-of-thought.

Step-by-Step Evaluation Process (Internal)
	1.	Read the Two Responses
	•	Carefully read both Response A and Response B.
	•	Internally identify key points, strengths, and potential shortcomings.
	2.	Summarize Each Response
	•	In your own words, create a concise summary of what each response is saying or proposing.
	3.	Evaluate Each Response Across these specific metrics: {', '.join(metrics)}
For each metric, assign a score from 1 (very poor) to 10 (excellent).
	•	Clarity: How clear, understandable, and well-expressed is the response?
	•	Completeness: To what extent does it address all aspects of the topic/question?
	•	Accuracy: How reliable, correct, and factually sound is the information provided?
	•	Reasoning Quality: Is the explanation or argument well-structured and logical?
	•	Practical Applicability: How useful or actionable is the response for real-world application?
	4.	Justify Each Score
	•	Provide a brief but sufficiently detailed explanation for why you gave the score on each metric.
	•	Highlight specific statements or reasoning steps within each response to illustrate your points.
	5.	Compare Strengths and Weaknesses
	•	Contrast the two responses across the five metrics.
	•	Note distinct advantages or disadvantages one response may have over the other.
	6.	Formulate an Overall Recommendation
	•	Decide which response is superior, or state if they are equally strong.
	•	Provide a concise rationale for your recommendation.
	7.	Prepare a Final, Structured Output
	•	Summarize your findings clearly for the user.
	•	Include only your final justifications (do not reveal your full chain-of-thought).

Required Output Format
	1.	Summaries of Both Responses
	•	A short, plain-language summary of Response A and Response B.
	2.	Scores and Justifications
	•	For each metric (Clarity, Completeness, Accuracy, Reasoning Quality, Practical Applicability), provide:
	•	Score for Response A with a brief explanation.
	•	Score for Response B with a brief explanation.
	3.	Strengths and Weaknesses Comparison
	•	A concise table or paragraph comparing the two responses, highlighting key strengths and weaknesses.
	4.	Overall Recommendation
	•	Which response is preferable (or whether they are equally strong), supported by a brief rationale.

Example of What the Judge LLM’s Final Output Might Look Like

	Summaries
Response A: Summarizes key points and overall conclusion.
Response B: Summarizes key points and overall conclusion.

	Scores and Justifications
		•	Clarity:
	•	Response A: 8/10 (Explanation)
	•	Response B: 7/10 (Explanation)
	•	Completeness:
	•	Response A: 9/10 (Explanation)
	•	Response B: 6/10 (Explanation)
	•	(Repeat for Accuracy, Reasoning Quality, Practical Applicability)

	Strengths and Weaknesses
		•	Response A excels in X. It could improve on Y.
	•	Response B provides Z but lacks detail on Q.

	Overall Recommendation
		•	e.g., “Response A is generally stronger due to better completeness and reasoning quality.”

Notes & Best Practices
	•	Keep Chain-of-Thought Internal: While you should use step-by-step reasoning to evaluate the responses thoroughly, do not reveal your entire thought process in the final answer. Summaries and succinct justifications are sufficient for the user.
	•	Maintain Impartiality: Provide unbiased, evidence-based assessments.
	•	Be Specific and Concrete: When justifying scores, point to actual content from the responses to illustrate your reasoning.
	•	Use Clear Language: The final output should be easily understandable to a broad range of users.
    """

def load_snapshot_previews(snapshot1_id: int, snapshot2_id: int, aspects: List[str]) -> Tuple[str, str]:
    """Load and format selected aspects of two snapshots for preview"""
    try:
        # Validate inputs
        if not snapshot1_id or not snapshot2_id:
            return "", ""
            
        # Get snapshots from database
        snap1 = db.get_snapshot_by_id(int(snapshot1_id))
        snap2 = db.get_snapshot_by_id(int(snapshot2_id))
        
        if not snap1 or not snap2:
            return "Snapshot not found", "Snapshot not found"
        
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
        
        return format_snapshot(snap1), format_snapshot(snap2)
        
    except Exception as e:
        return f"Error loading preview: {str(e)}", f"Error loading preview: {str(e)}"

def evaluate_snapshots(
    snapshot1_id: int,
    snapshot2_id: int,
    aspects: List[str],
    judge_model: str,
    metrics: List[str],
    custom_criteria: str
) -> Tuple[Dict, str]:
    """Evaluate two snapshots using the specified model and criteria"""
    try:
        # Validate inputs
        if not snapshot1_id or not snapshot2_id:
            return {}, "Please select both snapshots for comparison"
            
        # Get snapshots
        snap1 = db.get_snapshot_by_id(int(snapshot1_id))
        snap2 = db.get_snapshot_by_id(int(snapshot2_id))
        
        if not snap1 or not snap2:
            return {}, "One or both snapshots not found"
            
        # Prepare content for comparison
        content1 = ""
        content2 = ""
        
        for aspect in aspects:
            if aspect == "Thinking":
                content1 += f"=== Thinking ===\n{snap1.get('thinking', '')}\n\n"
                content2 += f"=== Thinking ===\n{snap2.get('thinking', '')}\n\n"
            elif aspect == "Reflection":
                content1 += f"=== Reflection ===\n{snap1.get('reflection', '')}\n\n"
                content2 += f"=== Reflection ===\n{snap2.get('reflection', '')}\n\n"
            elif aspect == "Final Output":
                content1 += f"=== Final Output ===\n{snap1.get('final_response', '')}\n\n"
                content2 += f"=== Final Output ===\n{snap2.get('final_response', '')}\n\n"
        
        # Create evaluation prompt
        evaluation_prompt = create_evaluation_prompt(
            content1=content1,
            content2=content2,
            metrics=metrics,
            custom_criteria=custom_criteria
        )
        
        # Get evaluation from model
        evaluation_response = get_model_response(judge_model, evaluation_prompt)
        
        # Parse numerical scores
        scores = {}
        for metric in metrics:
            metric_name = metric.split(" (")[0]
            match = re.search(f"{metric_name}[:\-]\s*(\d+)", evaluation_response)
            if match:
                scores[metric_name] = int(match.group(1))
        
        return scores, evaluation_response
        
    except Exception as e:
        return {}, f"Error during evaluation: {str(e)}"

# Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".small-font-table { font-size: 0.9em !important; }"
) as iface:
    # Add logo and title at the top
    with gr.Row():
        gr.Image(
            value="/Users/tomc/git/reflection/images/Linklaters.svg.png",
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
                    
                    # Make document upload optional and expandable
                    with gr.Accordion("Upload Document (Optional)", open=False):
                        file_input = gr.File(
                            label="Upload Document",
                            file_types=["pdf", "docx"],
                            type="binary"
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
            
            # JSON output
            json_output = gr.JSON(
                label="Snapshot Content",
                visible=False  # Initially hidden
            )
            
            operation_status = gr.Textbox(label="Status")

        # New Tab Implementation
        with gr.TabItem("Snapshot Evaluator"):
            # Section 1: Snapshots Table with smaller font
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
                elem_classes="small-font-table"  # Add custom CSS class
            )

            # Section 2: Comparison Setup
            with gr.Row():
                snapshot1_id = gr.Number(
                    label="First Snapshot ID",
                    precision=0,
                    minimum=None,  # Remove minimum constraint
                    value=None,  # Empty by default
                    scale=1
                )
                snapshot2_id = gr.Number(
                    label="Second Snapshot ID",
                    precision=0,
                    minimum=None,  # Remove minimum constraint
                    value=None,  # Empty by default
                    scale=1
                )
            
            # Model and Aspects side by side
            with gr.Row():
                with gr.Column(scale=1):
                    judge_model = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Judge Model",
                        value="Gemini 2.0 Flash"
                    )
                with gr.Column(scale=1):
                    comparison_aspects = gr.CheckboxGroup(
                        choices=["Thinking", "Reflection", "Final Output"],
                        label="Select Aspects to Compare",
                        value=["Final Output"]
                    )

            # Section 3: Preview (Side by Side)
            with gr.Row():
                with gr.Column():
                    preview1 = gr.TextArea(
                        label="First Snapshot Preview",
                        interactive=False
                    )
                with gr.Column():
                    preview2 = gr.TextArea(
                        label="Second Snapshot Preview",
                        interactive=False
                    )

            # Section 4: Evaluation Criteria
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
                    value=["Clarity (1-10)", "Completeness (1-10)"]
                )
                
                custom_criteria = gr.TextArea(
                    label="Custom Evaluation Instructions",
                    placeholder="Add your custom evaluation criteria here...",
                    value=default_evaluation_prompt
                )

            # Section 5: Evaluation Results
            with gr.Row():
                evaluate_btn = gr.Button("Evaluate", variant="primary")
                export_eval_btn = gr.Button("Export Evaluation", variant="secondary")
                save_eval_btn = gr.Button("Save Evaluation", variant="secondary")

            with gr.Row():
                qualitative_analysis = gr.TextArea(
                    label="Evaluation Results",
                    interactive=False,
                    show_copy_button=True,  # Enable one-click copy
                    scale=2  # Make it wider
                )

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

    # Preview update handler - triggers when aspects selection changes or IDs change
    def update_previews(id1, id2, aspects):
        if not id1 or not id2 or not aspects:
            return "", ""
        return load_snapshot_previews(id1, id2, aspects)

    # Connect the preview updates
    for component in [snapshot1_id, snapshot2_id, comparison_aspects]:
        component.change(
            fn=update_previews,
            inputs=[snapshot1_id, snapshot2_id, comparison_aspects],
            outputs=[preview1, preview2]
        )

    # Evaluation handler
    evaluate_btn.click(
        fn=lambda *args: (
            evaluate_snapshots(*args)[1]  # Only return the qualitative analysis
        ),
        inputs=[
            snapshot1_id,
            snapshot2_id,
            comparison_aspects,
            judge_model,
            predefined_metrics,
            custom_criteria
        ],
        outputs=qualitative_analysis
    )

    # Connect search box to table updates
    search_box_eval.change(
        fn=update_snapshots_table,
        inputs=[search_box_eval],
        outputs=[snapshots_table_eval]
    )

if __name__ == "__main__":
    iface.launch(share=False)
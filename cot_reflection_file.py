import re
from vertexai.generative_models import GenerativeModel
from anthropic import AnthropicVertex

# Initialize models
AVAILABLE_MODELS = {
    # "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Fash": "gemini-2.0-flash-exp",
    "Claude 3.5 Sonnet": "claude-3-5-sonnet@20240620"
}
project_id = "genai-sandbox-421407"
def get_model_response(model_name, prompt):
    """Helper function to get response from selected model"""
    try:
        if "Gemini" in model_name:
            model = GenerativeModel(AVAILABLE_MODELS[model_name])
            response = model.generate_content(prompt).text
        elif "Claude" in model_name:
            client = AnthropicVertex(project_id=project_id, region="us-east5")
            message = client.messages.create(
                model=AVAILABLE_MODELS[model_name],
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text
        return response
    except Exception as e:
        return f"Error with {model_name}: {str(e)}"

def cot_reflection(system_prompt, cot_prompt, question, document_content=None, model_name="Gemini 1.5 Pro"):
    """
    Perform chain-of-thought reflection using the specified model
    """
    try:
        # Format the prompts
        doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
        thinking_prompt = f"{system_prompt}\n\n{doc_content}{cot_prompt}\n\nQuestion: {question}\n\nThinking:"
        
        # Get thinking response using selected model
        thinking_response = get_model_response(model_name, thinking_prompt)
        thinking = f"<thinking>{thinking_response}</thinking>"
        
        # Format reflection prompt
        reflection_prompt = f"{system_prompt}\n\nInitial thinking: {thinking_response}\n\nReflect on this thinking process. What are the key assumptions? Are there any logical gaps or potential biases? How can the reasoning be improved?"
        
        # Get reflection using selected model
        reflection = get_model_response(model_name, reflection_prompt)
        
        # Format final output prompt
        final_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nInitial thinking: {thinking_response}\n\nReflection: {reflection}\n\nBased on this reflection, provide an improved final answer:"
        
        # Get final output using selected model
        output = get_model_response(model_name, final_prompt)
        
        return thinking, reflection, output
        
    except Exception as e:
        return f"Error: {str(e)}", "", ""

# Default prompts
system_prompt = """You are a helpful AI assistant. When answering questions, think carefully and break down your reasoning step by step."""

cot_prompt = """You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:

    1. Think through the problem step by step within the <thinking> tags.
    2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
    3. Make any necessary adjustments based on your reflection.
    4. Provide your final, concise answer within the <output> tags, taking into account your thinking and reflection.

    Important: The <thinking> and <reflection> sections are for your internal reasoning process. 
    The actual response to the query must be contained within the <output> tags, but should be informed by your thinking and reflection.

    Use the following format for your response:
    <thinking>
    [Your step-by-step reasoning goes here.]
    </thinking>
    <reflection>
    [Your reflection on your reasoning, checking for errors or improvements]
    </reflection>
    <output>
    [Your final, concise answer to the query, informed by your thinking and reflection. This is the part that will be shown to the user.]
    </output>
"""
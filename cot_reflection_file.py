from app_config import LLM_API_KEYS
from litellm import completion
import re
import os
from vertexai.generative_models import GenerativeModel
from anthropic import AnthropicVertex
from openai import AzureOpenAI

# Initialize models with their deployment names
AVAILABLE_MODELS = {
    "Gemini 2.0 Flash":     ("vertex_ai",       "vertex_ai/gemini-2.0-flash-exp",           "us-central1"),
    "Claude 3.5 Sonnet":    ("vertex_ai",       "vertex_ai/claude-3-5-sonnet@20240620",     "us-east5"),
    "Llama 3.1 70B":        ("azure_ai",        "azure_ai/llama-3-1-70b-instruct",          "https://api-llama-3-1-70b-instruct-live-llsyids.swedencentral.models.ai.azure.com/"),
    "Llama 3.1 405B":       ("azure_ai",        "azure_ai/llama-3-1-405b-instruct",         "https://api-llama-3-1-405b-instruct-live-llsyids.eastus.models.ai.azure.com/"),
    "Llama 3.3 70B":        ("azure_ai",        "azure_ai/llama-3-3-70b-instruct",          "https://api-llama-3-3-70b-instruct-teardown-llsyids.eastus.models.ai.azure.com/"),
    "OpenAI gpt-4o":        ("azure_ai",        "azure_ai/gpt-4o",                          "https://swedencentral.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview")
}

def get_model_response(model_name: str, prompt: str) -> str:
    """
    Helper function to get response from selected model
    
    Args:
        model_name: Name of the model to use
        prompt: Input prompt
        
    Returns:
        Generated text response
    """
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_provider, model_id, location_or_base = AVAILABLE_MODELS[model_name]
        
        if model_provider == "vertex_ai":
            response = completion(
                model=model_id,
                messages=[{"content": prompt, "role": "user"}],
                vertex_location=location_or_base
            )
        elif model_provider == "azure_ai":
            response = completion(
                model=model_id,
                messages=[{"content": prompt, "role": "user"}],
                api_key=LLM_API_KEYS[model_id],
                api_base=location_or_base
            )
        else:
            raise ValueError(f"Unknown provider: {model_provider}")
            
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error with {model_name}: {str(e)}"

def cot_reflection(
    system_prompt: str,
    cot_prompt: str,
    question: str,
    document_content: str = None,
    model_name: str = "Gemini 2.0 Flash"
) -> tuple[str, str, str]:
    """
    Perform chain-of-thought reflection using the specified model
    
    Args:
        system_prompt: System context prompt
        cot_prompt: Chain of thought prompt
        question: User question
        document_content: Optional document content
        model_name: Name of model to use
        
    Returns:
        Tuple of (thinking, reflection, output)
    """
    try:
        # Format the prompts
        doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
        thinking_prompt = f"{system_prompt}\n\n{doc_content}{cot_prompt}\n\nQuestion: {question}\n\nThinking:"
        
        # Get thinking response using selected model
        thinking_response = get_model_response(model_name, thinking_prompt)
        thinking = f"<thinking>{thinking_response}</thinking>"
        
        # Format reflection prompt
        reflection_prompt = (
            f"{system_prompt}\n\nInitial thinking: {thinking_response}\n\n"
            "Reflect on this thinking process. What are the key assumptions? "
            "Are there any logical gaps or potential biases? How can the reasoning be improved?"
        )
        
        # Get reflection using selected model
        reflection = get_model_response(model_name, reflection_prompt)
        
        # Format final output prompt
        final_prompt = (
            f"{system_prompt}\n\nQuestion: {question}\n\n"
            f"Initial thinking: {thinking_response}\n\n"
            f"Reflection: {reflection}\n\n"
            "Based on this reflection, provide an improved final answer:"
        )
        
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
from app_config import LLM_API_KEYS
from litellm import completion
import re
import os
from vertexai.generative_models import GenerativeModel
from anthropic import AnthropicVertex
from openai import AzureOpenAI

# Initialize models with their deployment names and parameter ranges
AVAILABLE_MODELS = {
    "Gemini 2.0 Flash": {
        "provider": "vertex_ai",
        "model_id": "vertex_ai/gemini-2.0-flash-001",
        # "location": "us-central1",
        "location": "europe-west1",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    },
    "Claude 3.5 Sonnet": {
        "provider": "vertex_ai",
        "model_id": "vertex_ai/claude-3-5-sonnet-v2@20241022",
        # "location": "us-east5",
        "location": "europe-west1",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    },
    # "Llama 3.1 70B": {
    #     "provider": "azure_ai",
    #     "model_id": "azure_ai/llama-3-1-70b-instruct",
    #     "location": "https://api-llama-3-1-70b-instruct-live-llsyids.swedencentral.models.ai.azure.com/",
    #     "temp_range": (0.0, 2.0),
    #     "top_p_range": (0.0, 1.0)
    # },
    "Llama 3.1 405B": {
        "provider": "azure_ai",
        "model_id": "azure_ai/llama-3-1-405b-instruct",
        "location": "https://api-llama-3-1-405b-instruct-live-llsyids.eastus.models.ai.azure.com/",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    },
    "Llama 3.3 70B": {
        "provider": "azure_ai",
        "model_id": "azure_ai/llama-3-3-70b-instruct",
        "location": "https://api-llama-3-3-70b-instruct-live-llsyids.eastus.models.ai.azure.com/",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    },
    "OpenAI gpt-4o": {
        "provider": "azure_ai",
        "model_id": "azure_ai/gpt-4o",
        "location": "https://swedencentral.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    },
    "OpenAI gpt-o1": {
        "provider": "azure_ai",
        "model_id": "azure_ai/openai-o1-genaiteam-swec-live",
        "location": "https://openai-genaiteam-swec-live.openai.azure.com",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    },
    "AI21 Jamba 1.5 Large": {
        "provider": "azure_ai",
        "model_id": "azure_ai/AI21-Jamba-1-5-Large",
        "location": "https://AI21-Jamba-1-5-Large-hfrke.eastus.models.ai.azure.com/",
        "temp_range": (0.0, 1.0),
        "top_p_range": (0.0, 1.0)
    }
}

def get_model_response(model_name: str, prompt: str, temperature: float = 0.7, top_p: float = 0.95) -> str:
    """
    Helper function to get response from selected model
    
    Args:
        model_name: Name of the model to use
        prompt: Input prompt
        temperature: Temperature parameter for response generation
        top_p: Top-p parameter for response generation
        
    Returns:
        Generated text response
    """
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_config = AVAILABLE_MODELS[model_name]
        
        if model_config["provider"] == "vertex_ai":
            response = completion(
                model=model_config["model_id"],
                messages=[{"content": prompt, "role": "user"}],
                vertex_location=model_config["location"],
                temperature=temperature,
                top_p=top_p
            )
        elif model_config["provider"] == "azure_ai":
            response = completion(
                model=model_config["model_id"],
                messages=[{"content": prompt, "role": "user"}],
                api_key=LLM_API_KEYS[model_config["model_id"]],
                api_base=model_config["location"],
                temperature=temperature,
                top_p=top_p
            )
        else:
            raise ValueError(f"Unknown provider: {model_config['provider']}")
            
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error with {model_name}: {str(e)}"

def get_model_params(model_name: str) -> dict:
    """Get parameter ranges for a specific model"""
    if model_name in AVAILABLE_MODELS:
        return {
            "temp_range": AVAILABLE_MODELS[model_name]["temp_range"],
            "top_p_range": AVAILABLE_MODELS[model_name]["top_p_range"]
        }
    return {"temp_range": (0.0, 1.0), "top_p_range": (0.0, 1.0)}

def cot_reflection(
    system_prompt: str,
    cot_prompt: str,
    question: str,
    document_content: str = None,
    model_name: str = "Gemini 2.0 Flash",
    temperature: float = 0.7,
    top_p: float = 0.95
) -> tuple[str, str, str]:
    """
    Perform chain-of-thought reflection using the specified model
    
    Args:
        system_prompt: System context prompt
        cot_prompt: Chain of thought prompt
        question: User question
        document_content: Optional document content
        model_name: Name of model to use
        temperature: Temperature parameter for response generation
        top_p: Top-p parameter for response generation
        
    Returns:
        Tuple of (thinking, reflection, output)
    """
    try:
        # Format the prompts
        doc_content = f"Document Content:\n{document_content}\n\n" if document_content else ""
        thinking_prompt = f"{system_prompt}\n\n{doc_content}{cot_prompt}\n\nQuestion: {question}\n\nThinking:"
        
        # Get thinking response using selected model with parameters
        thinking_response = get_model_response(model_name, thinking_prompt, temperature, top_p)
        thinking = f"<thinking>{thinking_response}</thinking>"
        
        # Format reflection prompt
        reflection_prompt = (
            f"{system_prompt}\n\nInitial thinking: {thinking_response}\n\n"
            "Reflect on this thinking process. What are the key assumptions? "
            "Are there any logical gaps or potential biases? How can the reasoning be improved?"
        )
        
        # Get reflection using selected model with parameters
        reflection = get_model_response(model_name, reflection_prompt, temperature, top_p)
        
        # Format final output prompt
        final_prompt = (
            f"{system_prompt}\n\nQuestion: {question}\n\n"
            f"Initial thinking: {thinking_response}\n\n"
            f"Reflection: {reflection}\n\n"
            "Based on this reflection, provide an improved final answer:"
        )
        
        # Get final output using selected model with parameters
        output = get_model_response(model_name, final_prompt, temperature, top_p)
        
        return thinking, reflection, output
        
    except Exception as e:
        return f"Error: {str(e)}", "", ""

# Default prompts
system_prompt = """You are a helpful AI assistant. When answering questions, think carefully and break down your reasoning step by step."""

cot_prompt = """You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these structured steps:

    1. THINKING PHASE:
       - Break down the problem into clear logical steps
       - Consider relevant context and assumptions
       - Identify key components of the question
       - Show your reasoning process step by step
       Use <thinking> tags for this phase.

    2. REFLECTION PHASE:
       - Review your thinking process critically
       - Check for potential biases or logical gaps
       - Consider alternative perspectives
       - Identify areas for improvement
       - Validate your assumptions
       Use <reflection> tags for this phase.

    3. OUTPUT PHASE:
       - Synthesize your thinking and reflection
       - Provide a clear, concise answer
       - Focus on actionable insights
       - Ensure completeness and accuracy
       Use <output> tags for this phase.

    IMPORTANT: 
    - The <thinking> and <reflection> sections are your internal reasoning process but should be shown to the user.
    - The <output> section is what will be shown to the user
    - Each section should be clearly separated and properly tagged
    - Maintain consistent tag usage throughout

    EXAMPLE FORMAT:
    <thinking>
    1. First, I understand that...
    2. The key aspects to consider are...
    3. Following this logic...
    4. This leads me to conclude...
    </thinking>

    <reflection>
    1. Reviewing my thinking:
       - Strengths: [list key valid points]
       - Potential gaps: [list areas needing attention]
    2. Additional considerations:
       - [list any missing perspectives]
    3. Validation:
       - [confirm or adjust initial conclusions]
    </reflection>

    <output>
    [Clear, concise, and complete answer incorporating insights from thinking and reflection phases]
    </output>
"""
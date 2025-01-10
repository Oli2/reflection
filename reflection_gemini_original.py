import warnings
warnings.filterwarnings("ignore")

from docx import Document
import argparse
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPICallError, InvalidArgument
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

# Function to read docx file
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Initialize Vertex AI and get Gemini Pro model
# def initialize_vertex_ai(project: str, location: str, credentials_path: str = None):
#     if credentials_path:
#         credentials = service_account.Credentials.from_service_account_file(credentials_path)
#         aiplatform.init(project=project, location=location, credentials=credentials)
#     else:
#         aiplatform.init(project=project, location=location)
#     return TextGenerationModel.from_pretrained("gemini-1.5-pro")





# Function to query Gemini Pro
def query_gemini_pro(model, prompt: str, return_full_response: bool = False):
    try:
        response = model.generate_content(contents=[prompt])
        return response.text
    except (GoogleAPICallError, InvalidArgument) as e:
        print(f"API call error: {e}")
        return None

# Define prompts
INITIAL_PROMPT_TEMPLATE = """
You are a legal assistant. Provide a detailed and accurate answer to the following question based on the content of the given document.

Document Content:
{document_content}

Question: {question}

Answer:
"""

REFLECTION_PROMPT_TEMPLATE = """
You are a senior legal expert reviewing the assistant's answer for correctness, completeness, and clarity.

Document Content:
{document_content}

Question: {question}

Assistant's Answer:
{initial_answer}

Provide specific feedback on any inaccuracies, omissions, or areas needing improvement.

Feedback:
"""

REFINEMENT_PROMPT_TEMPLATE = """
You are a legal assistant who has received feedback from a senior legal expert.

Document Content:
{document_content}

Question: {question}

Feedback:
{feedback}

Based on this feedback, revise your original answer to improve its accuracy, completeness, and clarity.

Original Answer:
{initial_answer}

Revised Answer:
"""

def main(docx_path, question, project, location, credentials_path):
    # Initialize Vertex AI and get Gemini Pro model
    MODEL_ID = "gemini-1.5-pro"
    model = GenerativeModel(MODEL_ID)
    
    # Read the document content
    document_content = read_docx(docx_path)
    
    # Generate Initial Answer
    initial_prompt = INITIAL_PROMPT_TEMPLATE.format(
        document_content=document_content,
        question=question
    )
    print("Generating Initial Answer...")
    initial_answer = query_gemini_pro(model, initial_prompt)
    if initial_answer is None:
        print("Failed to generate initial answer.")
        return
    
    # Generate Reflection
    reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
        document_content=document_content,
        question=question,
        initial_answer=initial_answer
    )
    print("Generating Feedback...")
    feedback = query_gemini_pro(model, reflection_prompt)
    if feedback is None:
        print("Failed to generate feedback.")
        return
    
    # Generate Revised Answer
    refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(
        document_content=document_content,
        question=question,
        feedback=feedback,
        initial_answer=initial_answer
    )
    print("Generating Revised Answer...")
    revised_answer = query_gemini_pro(model, refinement_prompt)
    if revised_answer is None:
        print("Failed to generate revised answer.")
        return
    
    # Output Results
    print("\n=== Analysis Results ===")
    print("\nQuestion:", question)
    print("\n--- Initial Answer ---\n", initial_answer)
    print("\n--- Feedback ---\n", feedback)
    print("\n--- Revised Answer ---\n", revised_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Word document using AI reflection with Vertex AI Gemini Pro.")
    parser.add_argument('--read', type=str, required=True, help='Path to the Word document to analyze')
    parser.add_argument('-q', '--question', type=str, default="Is this document about cooking?", 
                        help='Question to ask about the document')
    parser.add_argument('--project', type=str, default="genai-sandbox-421407", help='GCP Project ID')
    parser.add_argument('--location', type=str, default="europe-west2", help='GCP Location')
    parser.add_argument('--credentials', type=str, default="/Users/tomc/service_acccount_key.json", help='Path to GCP service account JSON key file')
    
    args = parser.parse_args()
    
    main(args.read, args.question, args.project, args.location, args.credentials)
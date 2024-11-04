import warnings
warnings.filterwarnings("ignore")

from docx import Document
import argparse
import logging
from abc import ABC, abstractmethod
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPICallError, InvalidArgument
from vertexai.generative_models import GenerativeModel
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load prompt templates from YAML file
with open('prompt_templates.yaml', 'r') as file:
    PROMPT_TEMPLATES = yaml.safe_load(file)

class ModelInterface(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

class GeminiInterface(ModelInterface):
    def __init__(self, model_name: str):
        self.model = GenerativeModel(model_name)

    def query(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(contents=[prompt])
            return response.text
        except (GoogleAPICallError, InvalidArgument) as e:
            logger.error(f"Gemini API call error: {e}")
            return None

class TextGenerationInterface(ModelInterface):
    def __init__(self, model_name: str):
        self.model = TextGenerationModel.from_pretrained(model_name)

    def query(self, prompt: str) -> str:
        try:
            response = self.model.predict(prompt=prompt, max_output_tokens=1024)
            return response.text
        except Exception as e:
            logger.error(f"Text Generation API call error: {e}")
            return None

def create_model_interface(model_name: str) -> ModelInterface:
    try:
        if 'gemini' in model_name.lower():
            return GeminiInterface(model_name)
        else:
            return TextGenerationInterface(model_name)
    except Exception as e:
        logger.warning(f"Failed to create interface for {model_name}: {e}")
        logger.info("Falling back to Gemini-1.5-pro")
        return GeminiInterface('gemini-1.5-pro')

def read_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error reading document: {e}")
        raise

def main(docx_path: str, question: str, model_name: str):
    try:
        model_interface = create_model_interface(model_name)
        document_content = read_docx(docx_path)

        initial_prompt = PROMPT_TEMPLATES['INITIAL_PROMPT_TEMPLATE'].format(
            document_content=document_content,
            question=question
        )
        logger.info("Generating Initial Answer...")
        initial_answer = model_interface.query(initial_prompt)
        if initial_answer is None:
            return "Failed to generate initial answer.", "", ""

        reflection_prompt = PROMPT_TEMPLATES['REFLECTION_PROMPT_TEMPLATE'].format(
            document_content=document_content,
            question=question,
            initial_answer=initial_answer
        )
        logger.info("Generating Feedback...")
        feedback = model_interface.query(reflection_prompt)
        if feedback is None:
            return initial_answer, "Failed to generate feedback.", ""

        refinement_prompt = PROMPT_TEMPLATES['REFINEMENT_PROMPT_TEMPLATE'].format(
            document_content=document_content,
            question=question,
            feedback=feedback,
            initial_answer=initial_answer
        )
        logger.info("Generating Revised Answer...")
        revised_answer = model_interface.query(refinement_prompt)
        if revised_answer is None:
            return initial_answer, feedback, "Failed to generate revised answer."

        return initial_answer, feedback, revised_answer
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        return f"An error occurred: {str(e)}", "", ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Word document using AI reflection with Vertex AI models.")
    parser.add_argument('--read', type=str, required=True, help='Path to the Word document to analyze')
    parser.add_argument('-q', '--question', type=str, default="Is this document about cooking?", 
                        help='Question to ask about the document')
    parser.add_argument('--model', type=str, choices=['gemini-1.5-pro', 'claude-3-5-sonnet-v2@20241022'], 
                        default='gemini-1.5-pro', help='Model to use for analysis')
    
    args = parser.parse_args()
    
    initial_answer, feedback, revised_answer = main(args.read, args.question, args.model)
    
    # Output Results
    print("\n=== Analysis Results ===")
    print("\nQuestion:", args.question)
    print("\n--- Initial Answer ---\n", initial_answer)
    print("\n--- Feedback ---\n", feedback)
    print("\n--- Revised Answer ---\n", revised_answer)

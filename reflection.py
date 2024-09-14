import warnings
warnings.filterwarnings("ignore")

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from docx import Document
import argparse
import os

# Function to read docx file
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4")

# Define the initial answer chain
initial_prompt = PromptTemplate(
    input_variables=["document_content", "question"],
    template="""
You are a legal assistant. Provide a detailed and accurate answer to the following question based on the content of the given document.

Document Content:
{document_content}

Question: {question}

Answer:"""
)
initial_chain = LLMChain(llm=llm, prompt=initial_prompt)

# Define the reflection chain
reflection_prompt = PromptTemplate(
    input_variables=["document_content", "question", "initial_answer"],
    template="""
You are a senior legal expert reviewing the assistant's answer for correctness, completeness, and clarity.

Document Content:
{document_content}

Question: {question}

Assistant's Answer:
{initial_answer}

Provide specific feedback on any inaccuracies, omissions, or areas needing improvement.

Feedback:"""
)
reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt)

# Define the refinement chain
refinement_prompt = PromptTemplate(
    input_variables=["document_content", "question", "initial_answer", "feedback"],
    template="""
You are a legal assistant who has received feedback from a senior legal expert.

Document Content:
{document_content}

Question: {question}

Feedback:
{feedback}

Based on this feedback, revise your original answer to improve its accuracy, completeness, and clarity.

Original Answer:
{initial_answer}

Revised Answer:"""
)
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)

def main(docx_path, question):
    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the script.")

    # Read the document content
    document_content = read_docx(docx_path)

    # Run the initial answer chain
    initial_result = initial_chain.invoke({"document_content": document_content, "question": question})
    
    # Run the reflection chain
    reflection_result = reflection_chain.invoke({"document_content": document_content, "question": question, "initial_answer": initial_result['text']})
    
    # Run the refinement chain
    final_result = refinement_chain.invoke({
        "document_content": document_content,
        "question": question,
        "initial_answer": initial_result['text'],
        "feedback": reflection_result['text']
    })

    print("Question:", question)
    print("\nInitial Answer:\n", initial_result['text'])
    print("\nFeedback:\n", reflection_result['text'])
    print("\nRevised Answer:\n", final_result['text'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Word document using AI reflection.")
    parser.add_argument('--read', type=str, required=True, help='Path to the Word document to analyze')
    parser.add_argument('-q', '--question', type=str, default="Is this document about cooking?", 
                        help='Question to ask about the document')
    
    args = parser.parse_args()
    
    main(args.read, args.question)
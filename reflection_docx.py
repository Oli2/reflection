from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
from docx import Document
import os
import argparse

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

Answer:""",
)
initial_chain = LLMChain(llm=llm, prompt=initial_prompt, output_key="initial_answer")

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

Feedback:""",
)
reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt, output_key="feedback")

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

Revised Answer:""",
)
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt, output_key="revised_answer")

# Create a sequential chain that runs the initial answer, reflection, and refinement
reflection_sequence = SequentialChain(
    chains=[initial_chain, reflection_chain, refinement_chain],
    input_variables=["document_content", "question"],
    output_variables=["initial_answer", "feedback", "revised_answer"],
)

# Add this new function at the end of the file
def main(docx_path):
    # Read the document content
    document_content = read_docx(docx_path)
    
    # Ask a question about the document
    question = "What are the key points discussed in this document?"

    # Use the call() method
    outputs = reflection_sequence({"document_content": document_content, "question": question})

    print("Initial Answer:\n", outputs["initial_answer"])
    print("\nFeedback:\n", outputs["feedback"])
    print("\nRevised Answer:\n", outputs["revised_answer"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Word document using AI reflection.")
    parser.add_argument('--read', type=str, required=True, help='Path to the Word document to analyze')
    
    args = parser.parse_args()
    
    main(args.read)
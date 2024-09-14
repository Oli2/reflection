# Legal Document Analysis with AI Reflection

This project uses AI-powered language models to analyze legal documents, answer questions, and provide refined responses through a process of reflection and refinement.

## Features

- Reads and analyzes Microsoft Word (.docx) documents
- Utilizes GPT-4 for initial analysis, reflection, and refinement of answers
- Implements a three-stage process: initial answer, expert reflection, and answer refinement
- Command-line interface for easy interaction

## Requirements

- Python 3.11+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/legal-document-analysis.git
   cd legal-document-analysis
   ```

2. Install the required packages:
   ```
   pip install langchain langchain_openai python-docx
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the script with the following command:

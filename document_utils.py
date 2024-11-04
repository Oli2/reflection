from docx import Document
import PyPDF2
import os

def read_document(file_path: str) -> str:
    """Read content from either DOCX or PDF files."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.docx':
            return read_docx(file_path)
        elif file_extension == '.pdf':
            return read_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise Exception(f"Error reading document: {str(e)}")

def read_docx(file_path: str) -> str:
    """Read content from DOCX file."""
    doc = Document(file_path)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def read_pdf(file_path: str) -> str:
    """Read content from PDF file."""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return '\n'.join(page.extract_text() for page in pdf_reader.pages) 
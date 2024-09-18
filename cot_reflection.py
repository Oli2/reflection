import re
import logging
import argparse
from vertexai.generative_models import GenerativeModel
from reflection_gemini import query_gemini_pro

logger = logging.getLogger(__name__)

def cot_reflection(system_prompt, question, return_full_response: bool=False):
    cot_prompt = f"""
        {system_prompt}

        You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:

        1. Think through the problem step by step within the <thinking> tags.
        2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
        3. Make any necessary adjustments based on your reflection.
        4. Provide your final, concise answer within the <output> tags.

        Important: The <thinking> and <reflection> sections are for your internal reasoning process only. 
        Do not include any part of the final answer in these sections. 
        The actual response to the query must be entirely contained within the <output> tags.

        Use the following format for your response:
        <thinking>
        [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
        <reflection>
        [Your reflection on your reasoning, checking for errors or improvements]
        </reflection>
        [Any adjustments to your thinking based on your reflection]
        </thinking>
        <output>
        [Your final, concise answer to the query. This is the only part that will be shown to the user.]
        </output>
        """
    combined_prompt = f"{cot_prompt}\n\nQuestion: {question}"
    # Make the API call
    MODEL_ID = "gemini-1.5-pro"
    model = GenerativeModel(MODEL_ID)
    response = query_gemini_pro(
        prompt=combined_prompt,
        model=model,
        return_full_response=return_full_response
    )

    # print(f"response: {response}")
    # Extract the full response
    full_response = response
    if full_response is None:
        print("Error: No response received from the API.")
        return None
    logger.info(f"CoT with Reflection :\n{full_response}")

    # Extract only the output if not returning full response
    # Use regex to extract the content within <thinking> and <output> tags
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."

    output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)
    output = output_match.group(1).strip() if output_match else full_response

    logger.info(f"Final output :\n{output}")

    if return_full_response:
        return full_response
    else:
        return output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply CoT using AI reflection with Vertex AI Gemini Pro.")
    # parser.add_argument('--read', type=str, required=True, help='Path to the Word document to analyze')
    parser.add_argument('-q', '--question', type=str, default="What country was the first victim of the Second World War?", 
                        help='Question to ask')
    parser.add_argument('--project', type=str, default="genai-sandbox-421407", help='GCP Project ID')
    parser.add_argument('--location', type=str, default="europe-west2", help='GCP Location')
    parser.add_argument('--credentials', type=str, default="/Users/tomc/service_acccount_key.json", help='Path to GCP service account JSON key file')
    parser.add_argument('--fullresponse', action='store_true', help='Return full response including chain-of-thought')
    args = parser.parse_args()
    system_prompt = """You are a legal assistant. Provide a detailed and accurate answer to the following question."""
    
    # print(f"args.question: {args.question}")
    result = cot_reflection(system_prompt=system_prompt, question=args.question, return_full_response= args.fullresponse)
    if result is not None:
        print(f"{'Full Response' if args.fullresponse else 'Final Answer'}:")
        print(result)
    else:
        print("Failed to get a valid response.")
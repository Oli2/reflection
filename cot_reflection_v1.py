import re
import logging
import argparse
from vertexai.generative_models import GenerativeModel
from reflection_gemini import query_gemini_pro

logger = logging.getLogger(__name__)

system_prompt = """You are a legal assistant. Provide a detailed and accurate answer to the following question."""

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

def cot_reflection(system_prompt, cot_prompt, question):
    combined_prompt = f"""
        {system_prompt}

        {cot_prompt}

        Question: {question}
    """

    # Make the API call
    MODEL_ID = "gemini-1.5-pro"
    model = GenerativeModel(MODEL_ID)
    response = query_gemini_pro(
        prompt=combined_prompt,
        model=model,
        return_full_response=True  # Always get full response
    )

    full_response = response
    if full_response is None:
        print("Error: No response received from the API.")
        return None, None, None
    logger.info(f"CoT with Reflection :\n{full_response}")

    # Extract thinking, reflection, and output
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
    reflection_match = re.search(r'<reflection>(.*?)</reflection>', full_response, re.DOTALL)
    output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)

    thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."
    reflection = reflection_match.group(1).strip() if reflection_match else "No reflection process provided."
    output = output_match.group(1).strip() if output_match else ""

    # If output is empty or not present, generate it using thinking and reflection
    if not output:
        output_prompt = f"""
        Based on the following thinking and reflection, provide a concise final answer to the question: "{question}"

        Thinking:
        {thinking}

        Reflection:
        {reflection}

        Final answer:
        """
        output = query_gemini_pro(
            prompt=output_prompt,
            model=model,
            return_full_response=False
        )

    logger.info(f"Final output :\n{output}")

    return thinking, reflection, output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply CoT using AI reflection with Vertex AI Gemini Pro.")
    parser.add_argument('-q', '--question', type=str, default="What country was the first victim of the Second World War?", 
                        help='Question to ask')
    parser.add_argument('--project', type=str, default="genai-sandbox-421407", help='GCP Project ID')
    parser.add_argument('--location', type=str, default="europe-west2", help='GCP Location')
    parser.add_argument('--credentials', type=str, default="/Users/tomc/service_acccount_key.json", help='Path to GCP service account JSON key file')
    args = parser.parse_args()

    thinking, reflection, output = cot_reflection(system_prompt=system_prompt, cot_prompt=cot_prompt, question=args.question)
    if thinking is not None and reflection is not None and output is not None:
        print("Thinking:")
        print(thinking)
        print("\nReflection:")
        print(reflection)
        print("\nFinal Answer:")
        print(output)
    else:
        print("Failed to get a valid response.")
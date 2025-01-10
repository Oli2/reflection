import os
from pathlib import Path

VERTEX_DEFAULT_CREDENTIALS_FILE_PATH = f"/Users/tomc/service_acccount_key.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_DEFAULT_CREDENTIALS_FILE_PATH

## API keys
LLM_API_KEYS={
    "azure_ai/llama-3-1-70b-instruct": "to be set",
    "azure_ai/llama-3-1-405b-instruct": "to be set",
    "azure_ai/llama-3-3-70b-instruct": "to be set",
    "azure_ai/gpt-4o": "to be set"
}
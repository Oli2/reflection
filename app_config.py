import os
from pathlib import Path

VERTEX_DEFAULT_CREDENTIALS_FILE_PATH = f"/Users/tomc/service_acccount_key.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_DEFAULT_CREDENTIALS_FILE_PATH

## API keys
LLM_API_KEYS={
    "azure_ai/llama-3-1-70b-instruct": "DM6Hk3tcc87Y0wJVu2rP0wSnPQl7hsiX",
    "azure_ai/llama-3-1-405b-instruct": "4zibPIJCvCg8dcXPTM5DaKPmUTM9dlbB",
    "azure_ai/llama-3-3-70b-instruct": "WHfCuNYnkphFAkH54h9b4g6OQLIXJom3",
    "azure_ai/gpt-4o": "1feef3866ab5485c910270d0ad659e93"
}
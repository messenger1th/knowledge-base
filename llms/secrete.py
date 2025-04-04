import json
import os
import sys

def get_api_key(key_name="deepseek-api-key") -> str:
    filepath = "llms/secrete.json"
    if not os.path.exists(filepath):
        sys.exit(f"❌ API key file '{filepath}' not found.")

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            sys.exit("❌ Invalid JSON format in secret file.")

    key = data.get(key_name)
    if not key:
        sys.exit(f"❌ API key '{key_name}' not found or empty in JSON.")
    return key
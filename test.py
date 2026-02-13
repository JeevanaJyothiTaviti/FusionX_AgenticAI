import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.models.list()
    print("✅ API Key is valid")
except Exception as e:
    print("❌ API Key failed")
    print(e)
